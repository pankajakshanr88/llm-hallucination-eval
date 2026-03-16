#!/usr/bin/env python
"""Test Judge-Human Agreement.

This evaluates the REFERENCE responses (from golden dataset) to see
if the LLM judge agrees with the human labels.

This answers: "Does our AI judge detect hallucinations the same way humans do?"

Usage:
    python scripts/run_judge_human_agreement.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult

from src.config import settings
from src.dataset import load_golden_dataset, HumanLabel
from src.evaluators import (
    FaithfulnessEvaluator,
    HallucinationDetector,
    AccuracyScorer,
)


# Initialize evaluators
faithfulness_eval = FaithfulnessEvaluator()
hallucination_eval = HallucinationDetector()
accuracy_eval = AccuracyScorer()


def create_reference_response_dataset(client: Client, dataset_name: str) -> str:
    """Create a dataset where we evaluate the REFERENCE responses."""
    
    # Delete existing dataset if it exists
    try:
        existing = list(client.list_datasets(dataset_name=dataset_name))
        if existing:
            client.delete_dataset(dataset_id=existing[0].id)
            print(f"  Deleted existing dataset: {dataset_name}")
    except Exception:
        pass
    
    # Load golden dataset
    golden = load_golden_dataset()
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Judge-Human Agreement Test - Evaluates reference responses",
    )
    
    print(f"  Created dataset: {dataset_name}")
    
    # Add examples - the KEY is that we put the REFERENCE response as the expected output
    for entry in golden:
        client.create_example(
            inputs={
                "question": entry.question,
                "contract_text": entry.contract_text,
                "contract_type": entry.contract_type,
                # Include reference response in inputs so evaluators can access it
                "reference_response": entry.assistant_response,
            },
            outputs={
                "human_label": entry.human_label.value,
                "human_hallucinated_claims": entry.hallucinated_claims,
                "human_accuracy_score": entry.accuracy_score,
                "human_faithfulness_score": entry.faithfulness_score,
            },
            metadata={
                "entry_id": entry.id,
                "question_category": entry.question_category.value,
                "reviewer": entry.reviewer_name,
            },
            dataset_id=dataset.id,
        )
    
    print(f"  Added {len(golden)} examples")
    return str(dataset.id)


def reference_response_target(inputs: dict) -> dict:
    """Target function that returns the REFERENCE response (not a new one).
    
    This is the key difference - we're evaluating the reference response,
    not generating a new one.
    """
    return {
        "response": inputs.get("reference_response", ""),
    }


def human_label_evaluator(run, example) -> EvaluationResult:
    """Show the human label for comparison."""
    outputs = example.outputs
    human_label = outputs.get("human_label", "unknown")
    
    # Convert to numeric for column display
    label_scores = {
        "faithful": 1.0,
        "partially_faithful": 0.5,
        "unfaithful": 0.0,
    }
    
    return EvaluationResult(
        key="human_label",
        score=label_scores.get(human_label, 0.0),
        comment=human_label,
    )


def judge_agrees_evaluator(run, example) -> EvaluationResult:
    """Check if judge agrees with human label."""
    inputs = example.inputs
    outputs = example.outputs
    
    human_label = outputs.get("human_label", "")
    reference_response = inputs.get("reference_response", "")
    question = inputs.get("question", "")
    contract_text = inputs.get("contract_text", "")
    
    # Run hallucination detector on reference response
    hallucination_result = hallucination_eval.detect(
        question=question,
        response=reference_response,
        contract_text=contract_text,
    )
    
    # Determine if judge agrees with human
    human_says_hallucinated = human_label in ("partially_faithful", "unfaithful")
    judge_says_hallucinated = hallucination_result.contains_hallucination
    
    agrees = human_says_hallucinated == judge_says_hallucinated
    
    return EvaluationResult(
        key="judge_agrees",
        score=1.0 if agrees else 0.0,
        comment=f"Human: {human_label}, Judge: {'hallucination' if judge_says_hallucinated else 'faithful'}",
        extra={
            "human_label": human_label,
            "judge_detected_hallucination": judge_says_hallucinated,
            "agreement": agrees,
        }
    )


def faithfulness_on_reference(run, example) -> EvaluationResult:
    """Evaluate faithfulness of the REFERENCE response."""
    inputs = example.inputs
    
    reference_response = inputs.get("reference_response", "")
    question = inputs.get("question", "")
    contract_text = inputs.get("contract_text", "")
    
    result = faithfulness_eval.evaluate(
        question=question,
        response=reference_response,
        contract_text=contract_text,
    )
    
    return EvaluationResult(
        key="judge_faithfulness",
        score=result.score,
        comment=result.label,
    )


def hallucination_on_reference(run, example) -> EvaluationResult:
    """Detect hallucinations in the REFERENCE response."""
    inputs = example.inputs
    
    reference_response = inputs.get("reference_response", "")
    question = inputs.get("question", "")
    contract_text = inputs.get("contract_text", "")
    
    result = hallucination_eval.detect(
        question=question,
        response=reference_response,
        contract_text=contract_text,
    )
    
    return EvaluationResult(
        key="judge_hallucination",
        score=result.score,
        comment=f"Detected: {result.contains_hallucination}, Severity: {result.severity}",
        extra={
            "detected": result.contains_hallucination,
            "severity": result.severity,
            "claims": [c.claim for c in result.hallucinated_claims],
        }
    )


def accuracy_on_reference(run, example) -> EvaluationResult:
    """Score accuracy of the REFERENCE response."""
    inputs = example.inputs
    
    reference_response = inputs.get("reference_response", "")
    question = inputs.get("question", "")
    contract_text = inputs.get("contract_text", "")
    
    result = accuracy_eval.score(
        question=question,
        response=reference_response,
        contract_text=contract_text,
    )
    
    return EvaluationResult(
        key="judge_accuracy",
        score=result.score,
        comment=result.reasoning[:100] if result.reasoning else "",
    )


def main():
    print("=" * 60)
    print("JUDGE-HUMAN AGREEMENT TEST")
    print("=" * 60)
    print()
    print("This evaluates the REFERENCE responses (not new ones)")
    print("to check if the LLM judge agrees with human labels.")
    print()
    
    # Validate
    errors = settings.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    
    client = Client()
    
    # Create dataset with reference responses
    dataset_name = "judge-human-agreement-test"
    print("Setting up dataset...")
    create_reference_response_dataset(client, dataset_name)
    print()
    
    experiment_name = f"judge-agreement-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"Running experiment: {experiment_name}")
    print("-" * 60)
    
    # Run evaluation
    results = evaluate(
        reference_response_target,  # Returns the reference response, not a new one
        data=dataset_name,
        evaluators=[
            human_label_evaluator,       # Shows human label (for comparison)
            judge_agrees_evaluator,      # Does judge agree with human?
            faithfulness_on_reference,   # Judge's faithfulness score
            hallucination_on_reference,  # Judge's hallucination detection
            accuracy_on_reference,       # Judge's accuracy score
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    # Calculate agreement rate
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    # Extract agreement scores
    agreement_scores = []
    for result in results:
        if hasattr(result, 'evaluation_results'):
            for eval_result in result.evaluation_results.get('results', []):
                if eval_result.key == 'judge_agrees':
                    agreement_scores.append(eval_result.score)
    
    if agreement_scores:
        agreement_rate = sum(agreement_scores) / len(agreement_scores)
        print(f"📊 JUDGE-HUMAN AGREEMENT RATE: {agreement_rate:.0%}")
        print(f"   Target: ≥80%")
        print(f"   Status: {'✅ PASS' if agreement_rate >= 0.80 else '❌ FAIL'}")
    
    print()
    print("=" * 60)
    print("VIEW IN LANGSMITH")
    print("=" * 60)
    print()
    print("Go to: Datasets & Experiments → judge-human-agreement-test")
    print()
    print("COLUMNS TO COMPARE:")
    print("  • human_label        → What human reviewer said (1.0=faithful, 0.5=partial, 0.0=unfaithful)")
    print("  • judge_agrees       → Does judge agree? (1.0=yes, 0.0=no)")
    print("  • judge_faithfulness → Judge's faithfulness score")
    print("  • judge_hallucination→ Judge's hallucination score")
    print("  • judge_accuracy     → Judge's accuracy score")
    print()
    print("Look for rows where judge_agrees = 0.0 to see disagreements!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

