#!/usr/bin/env python
"""
EVALUATION WITH DETAILED TOOLTIPS

The 'comment' field shows as a tooltip when you hover over scores in LangSmith.
This version includes full reasoning in the tooltips.

Usage:
    python scripts/run_with_tooltips.py
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


def create_dataset(client: Client, dataset_name: str) -> str:
    """Create dataset with all query types."""
    
    try:
        existing = list(client.list_datasets(dataset_name=dataset_name))
        if existing:
            client.delete_dataset(dataset_id=existing[0].id)
    except Exception:
        pass
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation with detailed tooltips on hover",
    )
    
    # Golden dataset
    golden = load_golden_dataset()
    for entry in golden:
        client.create_example(
            inputs={
                "question": entry.question,
                "contract_text": entry.contract_text,
                "contract_type": entry.contract_type,
                "reference_response": entry.assistant_response,
            },
            outputs={
                "human_label": entry.human_label.value,
                "source": "golden",
            },
            dataset_id=dataset.id,
        )
    
    # Uncovered queries
    uncovered = [
        {
            "question": "What is the software warranty period?",
            "contract_type": "Software License Agreement",
            "contract_text": "SOFTWARE LICENSE: 90-day warranty against defects. After 90 days, software provided as-is.",
        },
        {
            "question": "Can I cancel my subscription anytime?",
            "contract_type": "Subscription Agreement",
            "contract_text": "SUBSCRIPTION: Monthly at $49.99. Cancel anytime with 30 days notice. No refunds for partial months.",
        },
    ]
    
    for q in uncovered:
        resp = assistant(q)
        client.create_example(
            inputs={
                "question": q["question"],
                "contract_text": q["contract_text"],
                "contract_type": q["contract_type"],
                "reference_response": resp["response"],
            },
            outputs={"human_label": "not_labeled", "source": "uncovered"},
            dataset_id=dataset.id,
        )
    
    # Out of scope
    oos = [
        {"question": "Should I sign this contract?", "contract_type": "Lease", "contract_text": "LEASE..."},
        {"question": "Is this a good deal?", "contract_type": "Employment", "contract_text": "EMPLOYMENT..."},
    ]
    
    for q in oos:
        resp = assistant(q)
        client.create_example(
            inputs={
                "question": q["question"],
                "contract_text": q["contract_text"],
                "contract_type": q["contract_type"],
                "reference_response": resp["response"],
            },
            outputs={"human_label": "out_of_scope", "source": "out_of_scope"},
            dataset_id=dataset.id,
        )
    
    print(f"  Created dataset with {len(golden) + len(uncovered) + len(oos)} entries")
    return str(dataset.id)


def return_reference(inputs: dict) -> dict:
    return {"response": inputs.get("reference_response", "")}


# ============================================================
# EVALUATORS WITH DETAILED TOOLTIPS
# ============================================================

def col_accuracy(run, example) -> EvaluationResult:
    """Accuracy with full reasoning in tooltip."""
    inputs = example.inputs
    result = accuracy_eval.score(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    
    # Build detailed tooltip
    tooltip = f"""ACCURACY EVALUATION
─────────────────────
Score: {result.score:.0%}
Correctness: {result.correctness:.0%}
Completeness: {result.completeness:.0%}
Relevance: {result.relevance:.0%}
Precision: {result.precision:.0%}

REASONING:
{result.reasoning}

ERRORS FOUND:
{', '.join(result.errors) if result.errors else 'None'}"""
    
    return EvaluationResult(
        key="Accuracy",
        score=result.score,
        comment=tooltip,
    )


def col_faithfulness(run, example) -> EvaluationResult:
    """Faithfulness with full reasoning in tooltip."""
    inputs = example.inputs
    result = faithfulness_eval.evaluate(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    
    tooltip = f"""FAITHFULNESS EVALUATION
─────────────────────────
Score: {result.score:.0%}
Label: {result.label}

REASONING:
{result.reasoning}

UNSUPPORTED CLAIMS:
{chr(10).join('• ' + c for c in result.unsupported_claims) if result.unsupported_claims else 'None found'}"""
    
    return EvaluationResult(
        key="Faithfulness",
        score=result.score,
        comment=tooltip,
    )


def col_hallucination(run, example) -> EvaluationResult:
    """Hallucination with full reasoning in tooltip."""
    inputs = example.inputs
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    
    claims_text = ""
    if result.hallucinated_claims:
        for c in result.hallucinated_claims:
            claims_text += f"\n• [{c.severity.upper()}] {c.claim}\n  Reason: {c.reason}"
    else:
        claims_text = "None detected"
    
    tooltip = f"""HALLUCINATION DETECTION
─────────────────────────
Score: {result.score:.0%} (1.0 = clean)
Detected: {result.contains_hallucination}
Severity: {result.severity}
Count: {result.hallucination_count}

REASONING:
{result.reasoning}

HALLUCINATED CLAIMS:
{claims_text}"""
    
    return EvaluationResult(
        key="Hallucination",
        score=result.score,
        comment=tooltip,
    )


def col_coverage(run, example) -> EvaluationResult:
    """Coverage status with explanation."""
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
    
    tooltip = f"""COVERAGE STATUS
─────────────────
Status: {result.status.value}
Confidence: {result.confidence:.0%}
Similarity: {result.similarity_score:.0%}
Contract Type Match: {result.contract_type_match}

RECOMMENDATION:
{result.recommendation}"""
    
    return EvaluationResult(
        key="Coverage",
        score=scores.get(result.status, 0),
        comment=tooltip,
    )


def col_human_label(run, example) -> EvaluationResult:
    """Human label with explanation."""
    label = example.outputs.get("human_label", "not_labeled")
    source = example.outputs.get("source", "unknown")
    
    scores = {"faithful": 1.0, "partially_faithful": 0.5, "unfaithful": 0.0}
    score = scores.get(label, 0.5)
    
    tooltip = f"""HUMAN LABEL (Ground Truth)
────────────────────────────
Label: {label}
Source: {source}
Score: {score}

INTERPRETATION:
1.0 = Faithful (no issues)
0.5 = Partially faithful OR not labeled
0.0 = Unfaithful (contains hallucination)"""
    
    return EvaluationResult(
        key="Human_Label",
        score=score,
        comment=tooltip,
    )


def col_agreement(run, example) -> EvaluationResult:
    """Agreement with explanation."""
    inputs = example.inputs
    outputs = example.outputs
    
    human_label = outputs.get("human_label", "")
    source = outputs.get("source", "")
    
    if source != "golden":
        tooltip = """AGREEMENT: N/A
─────────────────
This query has no human label.
Agreement can only be calculated
for golden dataset entries."""
        return EvaluationResult(key="Agreement", score=0.5, comment=tooltip)
    
    result = hallucination_eval.detect(
        question=inputs.get("question", ""),
        response=inputs.get("reference_response", ""),
        contract_text=inputs.get("contract_text", ""),
    )
    
    human_says_bad = human_label in ("partially_faithful", "unfaithful")
    judge_says_bad = result.contains_hallucination
    agrees = human_says_bad == judge_says_bad
    
    tooltip = f"""JUDGE-HUMAN AGREEMENT
───────────────────────
Human Label: {human_label}
Human Says Hallucination: {human_says_bad}

Judge Detected: {result.contains_hallucination}
Judge Severity: {result.severity}

RESULT: {'✓ AGREE' if agrees else '✗ DISAGREE'}

{'Both agree the response is problematic.' if agrees and human_says_bad else ''}
{'Both agree the response is clean.' if agrees and not human_says_bad else ''}
{'Human found issues but judge missed them.' if not agrees and human_says_bad else ''}
{'Judge found issues but human said it was fine.' if not agrees and not human_says_bad else ''}"""
    
    return EvaluationResult(
        key="Agreement",
        score=1.0 if agrees else 0.0,
        comment=tooltip,
    )


def main():
    print()
    print("█" * 60)
    print("█" + "  EVALUATION WITH DETAILED TOOLTIPS  ".center(58) + "█")
    print("█" * 60)
    print()
    
    errors = settings.validate()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    
    client = Client()
    
    dataset_name = "hallucination-eval-tooltips"
    print("Creating dataset...")
    create_dataset(client, dataset_name)
    print()
    
    experiment_name = f"tooltips-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Running: {experiment_name}")
    print("-" * 60)
    
    results = evaluate(
        return_reference,
        data=dataset_name,
        evaluators=[
            col_coverage,
            col_human_label,
            col_accuracy,
            col_faithfulness,
            col_hallucination,
            col_agreement,
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"""
  Dataset: {dataset_name}
  
  HOVER OVER ANY SCORE to see detailed reasoning!
  
  Each cell tooltip shows:
  • Detailed scores breakdown
  • Full reasoning from the judge
  • Specific errors or claims found
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

