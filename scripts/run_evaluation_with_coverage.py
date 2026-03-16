#!/usr/bin/env python
"""Run evaluation with coverage metadata tagged in LangSmith.

This script:
1. Checks coverage status for each query
2. Tags each run with coverage_status metadata
3. Results can be filtered in LangSmith dashboard by coverage status

Usage:
    python scripts/run_evaluation_with_coverage.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

from src.config import settings
from src.dataset import load_golden_dataset, DatasetEntry
from src.coverage import CoverageDetector, CoverageStatus, CandidateQueue
from src.mock_assistant import MockDocumentAssistant, ResponseMode
from src.evaluators import (
    FaithfulnessEvaluator,
    HallucinationDetector,
    AccuracyScorer,
    GroundednessEvaluator,
)


# Initialize components
coverage_detector = CoverageDetector()
candidate_queue = CandidateQueue()

# Evaluators
faithfulness_eval = FaithfulnessEvaluator()
hallucination_eval = HallucinationDetector()
accuracy_eval = AccuracyScorer()
groundedness_eval = GroundednessEvaluator()


@traceable(
    name="document_assistant_evaluation",
    run_type="chain",
)
def evaluate_query_with_coverage(
    question: str,
    contract_text: str,
    contract_type: str,
    assistant: MockDocumentAssistant,
    golden_entry: Optional[DatasetEntry] = None,
) -> dict:
    """Evaluate a query with coverage metadata.
    
    This function is traced in LangSmith with coverage status as metadata.
    """
    # Check coverage status
    coverage_result = coverage_detector.detect(
        question=question,
        contract_type=contract_type,
        contract_text=contract_text,
    )
    
    # Get assistant response
    assistant_output = assistant({
        "question": question,
        "contract_text": contract_text,
        "contract_type": contract_type,
    })
    response = assistant_output.get("response", "")
    
    # Prepare result
    result = {
        "question": question,
        "contract_type": contract_type,
        "response": response,
        "coverage_status": coverage_result.status.value,
        "coverage_confidence": coverage_result.confidence,
        "similarity_score": coverage_result.similarity_score,
    }
    
    # Run appropriate evaluators based on coverage
    if coverage_result.status == CoverageStatus.GOLDEN_MATCH:
        # Full evaluation with golden reference
        faithfulness = faithfulness_eval.evaluate(question, response, contract_text)
        hallucination = hallucination_eval.detect(question, response, contract_text)
        accuracy = accuracy_eval.score(question, response, contract_text)
        
        result["evaluation_type"] = "full_golden"
        result["faithfulness_score"] = faithfulness.score
        result["faithfulness_label"] = faithfulness.label
        result["hallucination_score"] = hallucination.score
        result["hallucination_detected"] = hallucination.contains_hallucination
        result["hallucination_severity"] = hallucination.severity
        result["accuracy_score"] = accuracy.score
        
        if golden_entry:
            result["human_label"] = golden_entry.human_label.value
            result["matched_golden_id"] = golden_entry.id
            
    elif coverage_result.status == CoverageStatus.OUT_OF_SCOPE:
        # Out of scope - just check guardrail compliance
        groundedness = groundedness_eval.evaluate(question, response, contract_text)
        
        result["evaluation_type"] = "guardrail_check"
        result["guardrail_compliance"] = groundedness.guardrail_compliance
        result["appropriate_abstention"] = groundedness.appropriate_abstention
        
    else:
        # No coverage or partial - rubric only
        groundedness = groundedness_eval.evaluate(question, response, contract_text)
        
        result["evaluation_type"] = "rubric_only"
        result["groundedness_score"] = groundedness.groundedness
        result["relevance_score"] = groundedness.relevance
        result["clarity_score"] = groundedness.clarity
        result["tone_score"] = groundedness.tone
        
        # Log to candidate queue if not out of scope
        if coverage_result.status != CoverageStatus.OUT_OF_SCOPE:
            candidate_queue.add_candidate(
                question=question,
                contract_type=contract_type,
                contract_text=contract_text[:1000],
                assistant_response=response[:500],
                similarity_to_nearest=coverage_result.similarity_score,
                nearest_golden_id=coverage_result.matched_entry.id if coverage_result.matched_entry else None,
            )
    
    return result


def run_evaluation_with_coverage_tracking():
    """Run full evaluation with coverage tracking in LangSmith."""
    
    print("=" * 60)
    print("Evaluation with Coverage Tracking")
    print("=" * 60)
    print()
    
    # Validate config
    errors = settings.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    
    # Initialize
    client = Client()
    assistant = MockDocumentAssistant(mode=ResponseMode.LLM)
    golden_dataset = load_golden_dataset()
    
    print(f"Loaded {len(golden_dataset)} golden entries")
    print()
    
    # Test queries - mix of golden, uncovered, and out-of-scope
    test_queries = []
    
    # Add golden dataset entries
    for entry in golden_dataset:
        test_queries.append({
            "question": entry.question,
            "contract_type": entry.contract_type,
            "contract_text": entry.contract_text,
            "golden_entry": entry,
        })
    
    # Add some uncovered queries
    uncovered_queries = [
        {
            "question": "What is the software warranty period?",
            "contract_type": "Software License Agreement",
            "contract_text": "SOFTWARE LICENSE AGREEMENT\n\nThis agreement grants you a license to use the software...\n\n5. WARRANTY: The software is provided with a 90-day warranty against defects.",
            "golden_entry": None,
        },
        {
            "question": "Can I cancel my subscription anytime?",
            "contract_type": "Subscription Agreement", 
            "contract_text": "SUBSCRIPTION AGREEMENT\n\nTerm: Monthly subscription\n\nCancellation: You may cancel at any time with 30 days notice.",
            "golden_entry": None,
        },
    ]
    test_queries.extend(uncovered_queries)
    
    # Add out-of-scope queries
    out_of_scope_queries = [
        {
            "question": "Should I sign this contract?",
            "contract_type": "Residential Lease Agreement",
            "contract_text": "RESIDENTIAL LEASE...",
            "golden_entry": None,
        },
        {
            "question": "Is this a good deal for me?",
            "contract_type": "Employment Contract",
            "contract_text": "EMPLOYMENT AGREEMENT...",
            "golden_entry": None,
        },
    ]
    test_queries.extend(out_of_scope_queries)
    
    print(f"Total test queries: {len(test_queries)}")
    print(f"  - Golden entries: {len(golden_dataset)}")
    print(f"  - Uncovered queries: {len(uncovered_queries)}")
    print(f"  - Out-of-scope queries: {len(out_of_scope_queries)}")
    print()
    
    # Run evaluations
    print("Running evaluations...")
    print("-" * 60)
    
    results = {
        "golden_match": [],
        "partial_match": [],
        "no_coverage": [],
        "out_of_scope": [],
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query['question'][:50]}...")
        
        result = evaluate_query_with_coverage(
            question=query["question"],
            contract_text=query["contract_text"],
            contract_type=query["contract_type"],
            assistant=assistant,
            golden_entry=query.get("golden_entry"),
        )
        
        status = result["coverage_status"]
        results[status].append(result)
        
        print(f"    Coverage: {status}")
        print(f"    Eval Type: {result['evaluation_type']}")
        
        if "accuracy_score" in result:
            print(f"    Accuracy: {result['accuracy_score']:.0%}")
        if "groundedness_score" in result:
            print(f"    Groundedness: {result['groundedness_score']:.0%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(test_queries)
    
    print(f"\n📊 QUERY DISTRIBUTION:")
    print(f"   Golden Match:    {len(results['golden_match']):3d}  ({len(results['golden_match'])/total:5.1%})")
    print(f"   Partial Match:   {len(results['partial_match']):3d}  ({len(results['partial_match'])/total:5.1%})")
    print(f"   No Coverage:     {len(results['no_coverage']):3d}  ({len(results['no_coverage'])/total:5.1%})")
    print(f"   Out of Scope:    {len(results['out_of_scope']):3d}  ({len(results['out_of_scope'])/total:5.1%})")
    
    # Golden-only metrics
    if results["golden_match"]:
        golden = results["golden_match"]
        avg_accuracy = sum(r.get("accuracy_score", 0) for r in golden) / len(golden)
        avg_hallucination = sum(r.get("hallucination_score", 0) for r in golden) / len(golden)
        hallucination_rate = 1 - avg_hallucination
        
        print(f"\n📈 GOLDEN-MATCHED METRICS:")
        print(f"   Hallucination Rate: {hallucination_rate:5.1%}  (target: ≤5%)")
        print(f"   Accuracy:           {avg_accuracy:5.1%}  (target: ≥85%)")
    
    # Non-golden metrics
    non_golden = results["no_coverage"] + results["partial_match"]
    if non_golden:
        avg_groundedness = sum(r.get("groundedness_score", 0) for r in non_golden) / len(non_golden)
        print(f"\n📈 NON-GOLDEN METRICS (Rubric Only):")
        print(f"   Groundedness:       {avg_groundedness:5.1%}")
    
    # Candidate queue
    queue_stats = candidate_queue.get_stats()
    print(f"\n📝 CANDIDATE QUEUE:")
    print(f"   New candidates logged: {queue_stats['total_candidates']}")
    print(f"   Ready for review:      {queue_stats['ready_for_review']}")
    
    print("\n" + "=" * 60)
    print("VIEW IN LANGSMITH")
    print("=" * 60)
    print("""
To filter by coverage status in LangSmith:

1. Go to your project: document-assistant-hallucination-tracking
2. Click on "Runs" or "Traces"
3. Use the filter: 
   - metadata.coverage_status = "golden_match"
   - metadata.coverage_status = "no_coverage"
   - metadata.coverage_status = "out_of_scope"
   
4. Or filter by evaluation type:
   - metadata.evaluation_type = "full_golden"
   - metadata.evaluation_type = "rubric_only"
   - metadata.evaluation_type = "guardrail_check"
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation_with_coverage_tracking())

