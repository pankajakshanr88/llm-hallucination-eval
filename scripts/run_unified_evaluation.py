#!/usr/bin/env python
"""
UNIFIED EVALUATION - One Dataset, Everything You Need

This creates a SINGLE dataset with:
- Human labels (ground truth)
- Judge evaluations  
- Agreement status
- Coverage status

Usage:
    python scripts/run_unified_evaluation.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult

from src.config import settings
from src.dataset import load_golden_dataset, HumanLabel
from src.coverage import CoverageDetector, CoverageStatus
from src.evaluators import (
    FaithfulnessEvaluator,
    HallucinationDetector,
    AccuracyScorer,
)


# Initialize
coverage_detector = CoverageDetector()
faithfulness_eval = FaithfulnessEvaluator()
hallucination_eval = HallucinationDetector()
accuracy_eval = AccuracyScorer()


def create_unified_dataset(client: Client, dataset_name: str) -> str:
    """Create ONE unified dataset with all information."""
    
    # Delete existing
    try:
        existing = list(client.list_datasets(dataset_name=dataset_name))
        if existing:
            client.delete_dataset(dataset_id=existing[0].id)
            print(f"  Replaced existing dataset")
    except Exception:
        pass
    
    golden = load_golden_dataset()
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Unified hallucination evaluation - Human labels, Judge scores, Agreement",
    )
    
    for entry in golden:
        # Determine coverage
        coverage = coverage_detector.detect(
            question=entry.question,
            contract_type=entry.contract_type,
        )
        
        client.create_example(
            inputs={
                "question": entry.question,
                "contract_text": entry.contract_text,
                "contract_type": entry.contract_type,
                "reference_response": entry.assistant_response,
            },
            outputs={
                "human_label": entry.human_label.value,
                "hallucinated_claims": entry.hallucinated_claims,
            },
            metadata={
                "entry_id": entry.id,
                "category": entry.question_category.value,
                "coverage": coverage.status.value,
            },
            dataset_id=dataset.id,
        )
    
    print(f"  Created: {dataset_name} with {len(golden)} entries")
    return str(dataset.id)


def return_reference(inputs: dict) -> dict:
    """Return the reference response for evaluation."""
    return {"response": inputs.get("reference_response", "")}


# ============================================================
# EVALUATORS - Each becomes a column in the dashboard
# ============================================================

def human_label_column(run, example) -> EvaluationResult:
    """COLUMN: Human Label (ground truth)"""
    label = example.outputs.get("human_label", "unknown")
    scores = {"faithful": 1.0, "partially_faithful": 0.5, "unfaithful": 0.0}
    return EvaluationResult(
        key="1_Human_Label",
        score=scores.get(label, 0),
        comment=label,
    )


def judge_hallucination_column(run, example) -> EvaluationResult:
    """COLUMN: Judge's Hallucination Detection"""
    inputs = example.inputs
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    # Invert: 1.0 = no hallucination (faithful), 0.0 = hallucination detected
    return EvaluationResult(
        key="2_Judge_Score",
        score=result.score,
        comment=f"{'Hallucination' if result.contains_hallucination else 'Clean'}: {result.severity}",
    )


def agreement_column(run, example) -> EvaluationResult:
    """COLUMN: Does Judge Agree with Human?"""
    inputs = example.inputs
    outputs = example.outputs
    
    human_label = outputs.get("human_label", "")
    
    # Get judge's assessment
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    
    # Compare
    human_says_bad = human_label in ("partially_faithful", "unfaithful")
    judge_says_bad = result.contains_hallucination
    
    agrees = human_says_bad == judge_says_bad
    
    if agrees:
        comment = "✓ AGREE"
    else:
        comment = f"✗ DISAGREE (Human:{human_label}, Judge:{'halluc' if judge_says_bad else 'clean'})"
    
    return EvaluationResult(
        key="3_Agreement",
        score=1.0 if agrees else 0.0,
        comment=comment,
    )


def coverage_column(run, example) -> EvaluationResult:
    """COLUMN: Coverage Status"""
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
        key="4_Coverage",
        score=scores.get(result.status, 0),
        comment=result.status.value,
    )


def main():
    print()
    print("█" * 60)
    print("█" + "  UNIFIED HALLUCINATION EVALUATION  ".center(58) + "█")
    print("█" * 60)
    print()
    
    errors = settings.validate()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    
    client = Client()
    
    # ONE dataset
    dataset_name = "hallucination-evaluation"
    print("Creating unified dataset...")
    create_unified_dataset(client, dataset_name)
    print()
    
    experiment_name = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Running: {experiment_name}")
    print("-" * 60)
    
    results = evaluate(
        return_reference,
        data=dataset_name,
        evaluators=[
            human_label_column,        # 1. Human's label
            judge_hallucination_column, # 2. Judge's score
            agreement_column,          # 3. Do they agree?
            coverage_column,           # 4. Coverage status
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    # Calculate summary
    agreements = []
    for r in results:
        if hasattr(r, 'evaluation_results'):
            for er in r.evaluation_results.get('results', []):
                if er.key == '3_Agreement':
                    agreements.append(er.score)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if agreements:
        rate = sum(agreements) / len(agreements)
        print(f"""
  Total Samples:         {len(agreements)}
  Judge-Human Agreement: {rate:.0%}
  Target:                ≥80%
  Status:                {'✅ PASS' if rate >= 0.8 else '❌ FAIL'}
""")
    
    print("=" * 60)
    print("VIEW IN LANGSMITH")
    print("=" * 60)
    print(f"""
  Dataset: {dataset_name}
  
  COLUMNS:
  ┌──────────────┬─────────────────────────────────────┐
  │ 1_Human_Label│ Human's ground truth                │
  │              │ 1.0=faithful, 0.5=partial, 0=unfaith│
  ├──────────────┼─────────────────────────────────────┤
  │ 2_Judge_Score│ AI Judge's hallucination score      │
  │              │ 1.0=clean, 0.0=hallucination        │
  ├──────────────┼─────────────────────────────────────┤
  │ 3_Agreement  │ Do they match?                      │
  │              │ 1.0=AGREE, 0.0=DISAGREE             │
  ├──────────────┼─────────────────────────────────────┤
  │ 4_Coverage   │ Dataset coverage status             │
  │              │ 1.0=golden, 0.5=partial, 0=none     │
  └──────────────┴─────────────────────────────────────┘
  
  TIP: Sort by 3_Agreement to see disagreements at top!
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

