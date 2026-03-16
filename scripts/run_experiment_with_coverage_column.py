#!/usr/bin/env python
"""Run LangSmith experiment with coverage_status as a visible column.

This adds coverage_status to the experiment outputs so it appears
in the Datasets & Experiments view.

Usage:
    python scripts/run_experiment_with_coverage_column.py
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
)


# Initialize
coverage_detector = CoverageDetector()
faithfulness_eval = FaithfulnessEvaluator()
hallucination_eval = HallucinationDetector()
accuracy_eval = AccuracyScorer()


def coverage_status_evaluator(run, example) -> EvaluationResult:
    """Evaluator that returns coverage status as a score.
    
    This will appear as a column in the experiment view.
    """
    inputs = example.inputs
    
    result = coverage_detector.detect(
        question=inputs.get("question", ""),
        contract_type=inputs.get("contract_type", ""),
    )
    
    # Map status to numeric score for the column
    status_scores = {
        CoverageStatus.GOLDEN_MATCH: 1.0,
        CoverageStatus.PARTIAL_MATCH: 0.5,
        CoverageStatus.NO_COVERAGE: 0.0,
        CoverageStatus.OUT_OF_SCOPE: -1.0,
    }
    
    return EvaluationResult(
        key="coverage_status",
        score=status_scores.get(result.status, 0.0),
        comment=result.status.value,
        extra={
            "status": result.status.value,
            "similarity": result.similarity_score,
        }
    )


def evaluation_type_evaluator(run, example) -> EvaluationResult:
    """Evaluator that shows evaluation type (full vs rubric)."""
    inputs = example.inputs
    
    result = coverage_detector.detect(
        question=inputs.get("question", ""),
        contract_type=inputs.get("contract_type", ""),
    )
    
    if result.status == CoverageStatus.GOLDEN_MATCH:
        eval_type = "full_golden"
        score = 1.0
    elif result.status == CoverageStatus.OUT_OF_SCOPE:
        eval_type = "guardrail"
        score = 0.0
    else:
        eval_type = "rubric_only"
        score = 0.5
    
    return EvaluationResult(
        key="eval_type",
        score=score,
        comment=eval_type,
    )


def main():
    print("=" * 60)
    print("LangSmith Experiment with Coverage Status Column")
    print("=" * 60)
    print()
    
    # Validate
    errors = settings.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    
    client = Client()
    assistant = MockDocumentAssistant(mode=ResponseMode.LLM)
    
    dataset_name = "document-assistant-golden-dataset"
    experiment_name = f"coverage-experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {experiment_name}")
    print()
    print("Running evaluation...")
    print("-" * 60)
    
    # Run evaluation with coverage_status as an evaluator
    results = evaluate(
        assistant,
        data=dataset_name,
        evaluators=[
            coverage_status_evaluator,    # Shows coverage status
            evaluation_type_evaluator,    # Shows eval type
            faithfulness_eval,
            hallucination_eval,
            accuracy_eval,
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print()
    print("View in LangSmith:")
    print("  Datasets & Experiments → document-assistant-golden-dataset")
    print()
    print("NEW COLUMNS ADDED:")
    print("  • coverage_status  → 1.0 = golden_match, 0.5 = partial, 0.0 = no coverage")
    print("  • eval_type        → 1.0 = full_golden, 0.5 = rubric_only, 0.0 = guardrail")
    print()
    print("You can now sort/filter by these columns!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

