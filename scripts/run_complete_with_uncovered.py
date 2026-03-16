#!/usr/bin/env python
"""
COMPLETE EVALUATION WITH UNCOVERED QUERIES

Shows both:
- Coverage = 1.00 → Golden dataset queries (full evaluation)
- Coverage = 0.00 → Uncovered queries (rubric-only)
- Coverage = -1.00 → Out of scope (guardrail)

Usage:
    python scripts/run_complete_with_uncovered.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult

from src.config import settings
from src.dataset import load_golden_dataset
from src.coverage import CoverageDetector, CoverageStatus
from src.mock_assistant import MockDocumentAssistant, ResponseMode
from src.evaluators import (
    FaithfulnessEvaluator,
    HallucinationDetector,
    AccuracyScorer,
    GroundednessEvaluator,
)


# Initialize
coverage_detector = CoverageDetector()
faithfulness_eval = FaithfulnessEvaluator()
hallucination_eval = HallucinationDetector()
accuracy_eval = AccuracyScorer()
groundedness_eval = GroundednessEvaluator()
assistant = MockDocumentAssistant(mode=ResponseMode.LLM)


def create_mixed_dataset(client: Client, dataset_name: str) -> str:
    """Create dataset with BOTH golden and uncovered queries."""
    
    # Delete existing
    try:
        existing = list(client.list_datasets(dataset_name=dataset_name))
        if existing:
            client.delete_dataset(dataset_id=existing[0].id)
    except Exception:
        pass
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Complete evaluation with golden + uncovered + out-of-scope queries",
    )
    
    # 1. Add GOLDEN dataset entries
    golden = load_golden_dataset()
    for entry in golden:
        client.create_example(
            inputs={
                "question": entry.question,
                "contract_text": entry.contract_text,
                "contract_type": entry.contract_type,
                "has_reference": True,
                "reference_response": entry.assistant_response,
            },
            outputs={
                "human_label": entry.human_label.value,
                "source": "golden_dataset",
            },
            metadata={"entry_id": entry.id, "source": "golden"},
            dataset_id=dataset.id,
        )
    print(f"  Added {len(golden)} golden dataset entries")
    
    # 2. Add UNCOVERED queries (new contract types)
    uncovered_queries = [
        {
            "question": "What is the software warranty period?",
            "contract_type": "Software License Agreement",
            "contract_text": """SOFTWARE LICENSE AGREEMENT
            
1. LICENSE GRANT: We grant you a non-exclusive license to use this software.

2. WARRANTY: The software is provided with a 90-day warranty against defects.
   After 90 days, all warranties are void.

3. LIMITATIONS: The software is provided "as is" without any other warranties.

4. TERMINATION: This license terminates if you breach any terms.""",
        },
        {
            "question": "Can I cancel my subscription anytime?",
            "contract_type": "Subscription Agreement",
            "contract_text": """SUBSCRIPTION AGREEMENT

1. TERM: Monthly subscription, auto-renews each month.

2. CANCELLATION: You may cancel at any time with 30 days written notice.
   No refunds for partial months.

3. PAYMENT: $49.99/month, charged on the 1st of each month.

4. CHANGES: We may change pricing with 60 days notice.""",
        },
        {
            "question": "What are my obligations as a franchisee?",
            "contract_type": "Franchise Agreement",
            "contract_text": """FRANCHISE AGREEMENT

1. FRANCHISE FEE: Initial fee of $50,000 due upon signing.

2. ROYALTIES: 5% of gross sales, paid monthly.

3. OBLIGATIONS: Franchisee must maintain brand standards, use approved suppliers,
   and complete required training programs.

4. TERRITORY: Exclusive territory within 5-mile radius of location.""",
        },
    ]
    
    for q in uncovered_queries:
        # Generate a response for this query
        response = assistant({
            "question": q["question"],
            "contract_text": q["contract_text"],
            "contract_type": q["contract_type"],
        })
        
        client.create_example(
            inputs={
                "question": q["question"],
                "contract_text": q["contract_text"],
                "contract_type": q["contract_type"],
                "has_reference": False,
                "reference_response": response["response"],
            },
            outputs={
                "human_label": "not_labeled",
                "source": "uncovered_query",
            },
            metadata={"source": "uncovered"},
            dataset_id=dataset.id,
        )
    print(f"  Added {len(uncovered_queries)} uncovered queries")
    
    # 3. Add OUT-OF-SCOPE queries (legal advice)
    out_of_scope_queries = [
        {
            "question": "Should I sign this lease agreement?",
            "contract_type": "Residential Lease",
            "contract_text": "RESIDENTIAL LEASE AGREEMENT...",
        },
        {
            "question": "Is this employment contract a good deal for me?",
            "contract_type": "Employment Contract",
            "contract_text": "EMPLOYMENT AGREEMENT...",
        },
        {
            "question": "Can I sue my landlord based on this contract?",
            "contract_type": "Residential Lease",
            "contract_text": "RESIDENTIAL LEASE AGREEMENT...",
        },
    ]
    
    for q in out_of_scope_queries:
        response = assistant({
            "question": q["question"],
            "contract_text": q["contract_text"],
            "contract_type": q["contract_type"],
        })
        
        client.create_example(
            inputs={
                "question": q["question"],
                "contract_text": q["contract_text"],
                "contract_type": q["contract_type"],
                "has_reference": False,
                "reference_response": response["response"],
            },
            outputs={
                "human_label": "out_of_scope",
                "source": "out_of_scope_query",
            },
            metadata={"source": "out_of_scope"},
            dataset_id=dataset.id,
        )
    print(f"  Added {len(out_of_scope_queries)} out-of-scope queries")
    
    total = len(golden) + len(uncovered_queries) + len(out_of_scope_queries)
    print(f"  Total: {total} entries")
    
    return str(dataset.id)


def return_reference(inputs: dict) -> dict:
    """Return the reference response."""
    return {"response": inputs.get("reference_response", "")}


# ============================================================
# EVALUATORS
# ============================================================

def col_coverage(run, example) -> EvaluationResult:
    """Coverage Status - KEY COLUMN"""
    inputs = example.inputs
    result = coverage_detector.detect(
        question=inputs.get("question", ""),
        contract_type=inputs.get("contract_type", ""),
    )
    scores = {
        CoverageStatus.GOLDEN_MATCH: 1.0,
        CoverageStatus.PARTIAL_MATCH: 0.5,
        CoverageStatus.NO_COVERAGE: 0.0,
        CoverageStatus.OUT_OF_SCOPE: -1.0,
    }
    return EvaluationResult(
        key="Coverage",
        score=scores.get(result.status, 0),
        comment=result.status.value,
    )


def col_human_label(run, example) -> EvaluationResult:
    """Human Label (only for golden dataset)"""
    label = example.outputs.get("human_label", "not_labeled")
    scores = {
        "faithful": 1.0, 
        "partially_faithful": 0.5, 
        "unfaithful": 0.0,
        "not_labeled": None,
        "out_of_scope": None,
    }
    score = scores.get(label)
    return EvaluationResult(
        key="Human_Label",
        score=score if score is not None else 0.5,
        comment=label,
    )


def col_accuracy(run, example) -> EvaluationResult:
    """Accuracy Score"""
    inputs = example.inputs
    result = accuracy_eval.score(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    return EvaluationResult(
        key="Accuracy",
        score=result.score,
        comment=f"Score: {result.score:.0%}",
    )


def col_faithfulness(run, example) -> EvaluationResult:
    """Faithfulness Score"""
    inputs = example.inputs
    result = faithfulness_eval.evaluate(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    return EvaluationResult(
        key="Faithfulness",
        score=result.score,
        comment=result.label,
    )


def col_hallucination(run, example) -> EvaluationResult:
    """Hallucination Score (1.0 = clean, 0.0 = hallucination)"""
    inputs = example.inputs
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    return EvaluationResult(
        key="Hallucination",
        score=result.score,
        comment=f"{'HALLUCINATION' if result.contains_hallucination else 'Clean'}: {result.severity}",
    )


def col_agreement(run, example) -> EvaluationResult:
    """Agreement - only meaningful for golden dataset"""
    inputs = example.inputs
    outputs = example.outputs
    
    human_label = outputs.get("human_label", "")
    source = outputs.get("source", "")
    
    # Only calculate agreement for golden dataset
    if source != "golden_dataset":
        return EvaluationResult(
            key="Agreement",
            score=0.5,  # N/A shown as 0.5
            comment="N/A (no human label)",
        )
    
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    
    human_says_bad = human_label in ("partially_faithful", "unfaithful")
    judge_says_bad = result.contains_hallucination
    agrees = human_says_bad == judge_says_bad
    
    return EvaluationResult(
        key="Agreement",
        score=1.0 if agrees else 0.0,
        comment="✓ AGREE" if agrees else "✗ DISAGREE",
    )


def col_groundedness(run, example) -> EvaluationResult:
    """Groundedness - useful for uncovered queries"""
    inputs = example.inputs
    result = groundedness_eval.evaluate(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    return EvaluationResult(
        key="Groundedness",
        score=result.overall_score,
        comment=f"Grounded: {result.groundedness:.0%}",
    )


def main():
    print()
    print("█" * 60)
    print("█" + "  COMPLETE EVALUATION WITH ALL QUERY TYPES  ".center(58) + "█")
    print("█" * 60)
    print()
    
    errors = settings.validate()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    
    client = Client()
    
    dataset_name = "hallucination-tracking-full"
    print("Creating mixed dataset...")
    create_mixed_dataset(client, dataset_name)
    print()
    
    experiment_name = f"full-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Running: {experiment_name}")
    print("-" * 60)
    
    results = evaluate(
        return_reference,
        data=dataset_name,
        evaluators=[
            col_coverage,        # Coverage status (KEY!)
            col_human_label,     # Human label (golden only)
            col_accuracy,        # Accuracy
            col_faithfulness,    # Faithfulness
            col_hallucination,   # Hallucination
            col_agreement,       # Agreement (golden only)
            col_groundedness,    # Groundedness (all)
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    print()
    print("=" * 60)
    print("VIEW IN LANGSMITH")
    print("=" * 60)
    print(f"""
  Dataset: {dataset_name}
  
  COVERAGE VALUES:
  ┌─────────┬───────────────────────────────────────┐
  │  1.00   │ Golden dataset (full eval + agreement)│
  │  0.00   │ Uncovered query (rubric only)         │
  │ -1.00   │ Out of scope (guardrail check)        │
  └─────────┴───────────────────────────────────────┘
  
  SORT BY COVERAGE to see different query types!
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

