#!/usr/bin/env python
"""Run evaluation with coverage status as TAGS in LangSmith.

Tags are visible and filterable in the LangSmith dashboard.

Usage:
    python scripts/run_evaluation_with_tags.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client, traceable
from langchain_openai import ChatOpenAI

from src.config import settings
from src.dataset import load_golden_dataset, DatasetEntry
from src.coverage import CoverageDetector, CoverageStatus
from src.mock_assistant import MockDocumentAssistant, ResponseMode


def run_with_coverage_tags():
    """Run evaluation with coverage tags visible in LangSmith."""
    
    print("=" * 60)
    print("Evaluation with Coverage TAGS in LangSmith")
    print("=" * 60)
    print()
    
    # Validate config
    errors = settings.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    
    client = Client()
    assistant = MockDocumentAssistant(mode=ResponseMode.LLM)
    golden_dataset = load_golden_dataset()
    coverage_detector = CoverageDetector()
    
    # Test queries
    test_queries = []
    
    # Golden dataset entries
    for entry in golden_dataset:
        test_queries.append({
            "question": entry.question,
            "contract_type": entry.contract_type,
            "contract_text": entry.contract_text,
            "source": "golden",
        })
    
    # Uncovered queries
    test_queries.extend([
        {
            "question": "What is the software warranty period?",
            "contract_type": "Software License Agreement",
            "contract_text": "SOFTWARE LICENSE: 90-day warranty against defects.",
            "source": "uncovered",
        },
        {
            "question": "Can I cancel my subscription anytime?",
            "contract_type": "Subscription Agreement",
            "contract_text": "Cancel anytime with 30 days notice.",
            "source": "uncovered",
        },
    ])
    
    # Out-of-scope queries
    test_queries.extend([
        {
            "question": "Should I sign this contract?",
            "contract_type": "Residential Lease",
            "contract_text": "LEASE AGREEMENT...",
            "source": "out_of_scope",
        },
        {
            "question": "Is this a good deal for me?",
            "contract_type": "Employment Contract",
            "contract_text": "EMPLOYMENT AGREEMENT...",
            "source": "out_of_scope",
        },
    ])
    
    print(f"Running {len(test_queries)} queries with coverage tags...")
    print()
    
    results_by_status = {
        "golden_match": [],
        "no_coverage": [],
        "out_of_scope": [],
        "partial_match": [],
    }
    
    for i, query in enumerate(test_queries, 1):
        # Detect coverage
        coverage = coverage_detector.detect(
            question=query["question"],
            contract_type=query["contract_type"],
        )
        
        # Determine tags
        coverage_tag = f"coverage:{coverage.status.value}"
        eval_type_tag = "eval:full" if coverage.status == CoverageStatus.GOLDEN_MATCH else "eval:rubric_only"
        if coverage.status == CoverageStatus.OUT_OF_SCOPE:
            eval_type_tag = "eval:guardrail"
        
        tags = [coverage_tag, eval_type_tag, f"contract:{query['contract_type'][:20]}"]
        
        # Create a traced run with tags
        @traceable(
            name="document_assistant_query",
            tags=tags,
            metadata={
                "coverage_status": coverage.status.value,
                "evaluation_type": eval_type_tag.split(":")[1],
                "contract_type": query["contract_type"],
                "similarity_score": coverage.similarity_score,
            }
        )
        def run_query(question: str, contract_text: str, contract_type: str):
            # Get assistant response
            response = assistant({
                "question": question,
                "contract_text": contract_text,
                "contract_type": contract_type,
            })
            return response
        
        print(f"[{i}/{len(test_queries)}] {query['question'][:40]}...")
        print(f"    Tags: {tags}")
        
        result = run_query(
            question=query["question"],
            contract_text=query["contract_text"],
            contract_type=query["contract_type"],
        )
        
        results_by_status[coverage.status.value].append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE - VIEW IN LANGSMITH")
    print("=" * 60)
    print()
    print("Go to: Tracing Projects → document-assistant-hallucination-tracking")
    print()
    print("FILTER BY TAGS:")
    print("─" * 40)
    print("  coverage:golden_match  → Golden dataset queries")
    print("  coverage:no_coverage   → Uncovered queries")
    print("  coverage:out_of_scope  → Legal advice queries")
    print()
    print("  eval:full              → Full hallucination evaluation")
    print("  eval:rubric_only       → Groundedness check only")
    print("  eval:guardrail         → Guardrail response check")
    print()
    print("RESULTS SUMMARY:")
    print("─" * 40)
    for status, results in results_by_status.items():
        print(f"  {status}: {len(results)} queries")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_with_coverage_tags())

