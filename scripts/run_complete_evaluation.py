#!/usr/bin/env python
"""
COMPLETE UNIFIED EVALUATION - All Metrics in One Dataset

Includes:
- Human Label (ground truth)
- Accuracy (judge score)
- Faithfulness (judge score)
- Hallucination (judge score)
- Agreement (judge vs human)
- Coverage status

Usage:
    python scripts/run_complete_evaluation.py
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


# Initialize all evaluators
coverage_detector = CoverageDetector()
faithfulness_eval = FaithfulnessEvaluator()
hallucination_eval = HallucinationDetector()
accuracy_eval = AccuracyScorer()


def create_complete_dataset(client: Client, dataset_name: str) -> str:
    """Create complete dataset with all information."""
    
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
        description="Complete Hallucination Evaluation - All metrics in one place",
    )
    
    for entry in golden:
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
                "human_notes": entry.human_notes,
            },
            metadata={
                "entry_id": entry.id,
                "category": entry.question_category.value,
                "coverage": coverage.status.value,
                "reviewer": entry.reviewer_name,
            },
            dataset_id=dataset.id,
        )
    
    print(f"  Created: {dataset_name} with {len(golden)} entries")
    return str(dataset.id)


def return_reference(inputs: dict) -> dict:
    """Return the reference response for evaluation."""
    return {"response": inputs.get("reference_response", "")}


# ============================================================
# ALL EVALUATORS - Each becomes a column
# ============================================================

def col_human_label(run, example) -> EvaluationResult:
    """Human Label (Ground Truth)"""
    label = example.outputs.get("human_label", "unknown")
    scores = {"faithful": 1.0, "partially_faithful": 0.5, "unfaithful": 0.0}
    return EvaluationResult(
        key="Human_Label",
        score=scores.get(label, 0),
        comment=label,
    )


def col_accuracy(run, example) -> EvaluationResult:
    """Accuracy Score (Judge)"""
    inputs = example.inputs
    result = accuracy_eval.score(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    return EvaluationResult(
        key="Accuracy",
        score=result.score,
        comment=f"Correctness:{result.correctness:.0%} Completeness:{result.completeness:.0%}",
    )


def col_faithfulness(run, example) -> EvaluationResult:
    """Faithfulness Score (Judge)"""
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
    """Hallucination Score (Judge) - Higher = Better (no hallucination)"""
    inputs = example.inputs
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    return EvaluationResult(
        key="Hallucination",
        score=result.score,
        comment=f"Detected:{result.contains_hallucination} Severity:{result.severity}",
        extra={
            "detected": result.contains_hallucination,
            "severity": result.severity,
            "claims": [c.claim for c in result.hallucinated_claims],
        }
    )


def col_agreement(run, example) -> EvaluationResult:
    """Does Judge Agree with Human?"""
    inputs = example.inputs
    outputs = example.outputs
    
    human_label = outputs.get("human_label", "")
    
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
        comment="✓ AGREE" if agrees else f"✗ Human:{human_label}, Judge:{'bad' if judge_says_bad else 'good'}",
    )


def col_coverage(run, example) -> EvaluationResult:
    """Coverage Status"""
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


def main():
    print()
    print("█" * 60)
    print("█" + "  COMPLETE HALLUCINATION EVALUATION  ".center(58) + "█")
    print("█" + "  All Metrics in One Dataset  ".center(58) + "█")
    print("█" * 60)
    print()
    
    errors = settings.validate()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    
    client = Client()
    
    dataset_name = "hallucination-tracking-complete"
    print("Creating complete dataset...")
    create_complete_dataset(client, dataset_name)
    print()
    
    experiment_name = f"complete-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Running: {experiment_name}")
    print("-" * 60)
    
    results = evaluate(
        return_reference,
        data=dataset_name,
        evaluators=[
            col_human_label,      # Human ground truth
            col_accuracy,         # Judge: Accuracy
            col_faithfulness,     # Judge: Faithfulness
            col_hallucination,    # Judge: Hallucination
            col_agreement,        # Do they match?
            col_coverage,         # Coverage status
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    # Calculate summaries
    accuracy_scores = []
    faithfulness_scores = []
    hallucination_scores = []
    agreement_scores = []
    
    for r in results:
        if hasattr(r, 'evaluation_results'):
            for er in r.evaluation_results.get('results', []):
                if er.key == 'Accuracy':
                    accuracy_scores.append(er.score or 0)
                elif er.key == 'Faithfulness':
                    faithfulness_scores.append(er.score or 0)
                elif er.key == 'Hallucination':
                    hallucination_scores.append(er.score or 0)
                elif er.key == 'Agreement':
                    agreement_scores.append(er.score or 0)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    n = len(accuracy_scores) or 1
    avg_accuracy = sum(accuracy_scores) / n
    avg_faithfulness = sum(faithfulness_scores) / n
    avg_hallucination = sum(hallucination_scores) / n
    hallucination_rate = 1 - avg_hallucination
    agreement_rate = sum(agreement_scores) / n if agreement_scores else 0
    
    print(f"""
┌────────────────────────────┬──────────┬──────────┬────────┐
│ Metric                     │  Score   │  Target  │ Status │
├────────────────────────────┼──────────┼──────────┼────────┤
│ Accuracy                   │  {avg_accuracy:5.1%}  │   ≥85%  │ {'✅' if avg_accuracy >= 0.85 else '❌'}     │
│ Faithfulness               │  {avg_faithfulness:5.1%}  │   ≥85%  │ {'✅' if avg_faithfulness >= 0.85 else '❌'}     │
│ Hallucination Rate         │  {hallucination_rate:5.1%}  │   ≤5%   │ {'✅' if hallucination_rate <= 0.05 else '❌'}     │
│ Judge-Human Agreement      │  {agreement_rate:5.1%}  │   ≥80%  │ {'✅' if agreement_rate >= 0.80 else '❌'}     │
└────────────────────────────┴──────────┴──────────┴────────┘
""")
    
    print("=" * 60)
    print("VIEW IN LANGSMITH")
    print("=" * 60)
    print(f"""
  Dataset: {dataset_name}
  
  COLUMNS:
  ┌───────────────┬────────────────────────────────────────┐
  │ Human_Label   │ Ground truth (1=faithful, 0=unfaithful)│
  │ Accuracy      │ Judge's accuracy score                 │
  │ Faithfulness  │ Judge's faithfulness score             │
  │ Hallucination │ Judge's hallucination score (1=clean)  │
  │ Agreement     │ Judge matches human? (1=yes, 0=no)     │
  │ Coverage      │ Dataset coverage status                │
  └───────────────┴────────────────────────────────────────┘
  
  TIPS:
  • Sort by Agreement to see disagreements
  • Sort by Hallucination to see worst cases
  • All scores: higher = better
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

