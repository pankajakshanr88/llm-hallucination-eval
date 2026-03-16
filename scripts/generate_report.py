#!/usr/bin/env python
"""Generate reports from evaluation results.

Usage:
    python scripts/generate_report.py
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.pipeline import EvaluationResult, EvaluationMetrics


@dataclass
class EvaluationReport:
    """Structured evaluation report."""
    
    # Metadata
    timestamp: str
    total_samples: int
    
    # Core metrics
    hallucination_rate: float
    avg_accuracy: float
    avg_faithfulness: float
    judge_human_agreement: float
    
    # Thresholds
    hallucination_threshold: float
    accuracy_threshold: float
    agreement_threshold: float
    
    # Status
    passes_all_thresholds: bool
    
    # Details
    hallucinated_samples: list[dict] = field(default_factory=list)
    low_accuracy_samples: list[dict] = field(default_factory=list)
    disagreement_samples: list[dict] = field(default_factory=list)
    
    # Category breakdown
    metrics_by_category: dict = field(default_factory=dict)


def generate_report(
    results: list[EvaluationResult],
    metrics: EvaluationMetrics,
) -> EvaluationReport:
    """Generate a structured report from evaluation results.
    
    Args:
        results: List of individual evaluation results
        metrics: Aggregated metrics
        
    Returns:
        EvaluationReport with all details
    """
    # Find hallucinated samples
    hallucinated = []
    for r in results:
        if r.hallucination_detected:
            hallucinated.append({
                "id": r.entry_id,
                "question": r.question,
                "contract_type": r.contract_type,
                "claims": r.hallucinated_claims_detected,
                "faithfulness_score": r.faithfulness_score,
            })
    
    # Find low accuracy samples
    low_accuracy = []
    for r in results:
        if r.accuracy_score < settings.accuracy_threshold:
            low_accuracy.append({
                "id": r.entry_id,
                "question": r.question,
                "accuracy_score": r.accuracy_score,
            })
    
    # Find disagreements between judge and human
    disagreements = []
    for r in results:
        if not r.judge_agrees_with_human:
            disagreements.append({
                "id": r.entry_id,
                "question": r.question,
                "human_label": r.human_label,
                "judge_detected_hallucination": r.hallucination_detected,
            })
    
    return EvaluationReport(
        timestamp=datetime.now().isoformat(),
        total_samples=metrics.total_samples,
        hallucination_rate=metrics.hallucination_rate,
        avg_accuracy=metrics.avg_accuracy_score,
        avg_faithfulness=metrics.avg_faithfulness_score,
        judge_human_agreement=metrics.judge_human_agreement,
        hallucination_threshold=settings.hallucination_rate_threshold,
        accuracy_threshold=settings.accuracy_threshold,
        agreement_threshold=settings.judge_agreement_threshold,
        passes_all_thresholds=metrics.passes_hallucination_threshold and metrics.passes_accuracy_threshold,
        hallucinated_samples=hallucinated,
        low_accuracy_samples=low_accuracy,
        disagreement_samples=disagreements,
        metrics_by_category=metrics.metrics_by_category,
    )


def print_report(report: EvaluationReport) -> None:
    """Print a formatted report to stdout."""
    print()
    print("=" * 60)
    print("HALLUCINATION EVALUATION REPORT")
    print("=" * 60)
    print(f"Generated: {report.timestamp}")
    print(f"Total Samples: {report.total_samples}")
    print()
    
    # Summary metrics
    print("-" * 60)
    print("SUMMARY METRICS")
    print("-" * 60)
    print()
    
    # Hallucination rate with status indicator
    hall_status = "PASS" if report.hallucination_rate <= report.hallucination_threshold else "FAIL"
    print(f"  Hallucination Rate:     {report.hallucination_rate:>6.1%}  (threshold: <={report.hallucination_threshold:.0%}) [{hall_status}]")
    
    # Accuracy with status indicator
    acc_status = "PASS" if report.avg_accuracy >= report.accuracy_threshold else "FAIL"
    print(f"  Average Accuracy:       {report.avg_accuracy:>6.1%}  (threshold: >={report.accuracy_threshold:.0%}) [{acc_status}]")
    
    # Faithfulness
    print(f"  Average Faithfulness:   {report.avg_faithfulness:>6.1%}")
    
    # Judge-Human agreement
    agree_status = "PASS" if report.judge_human_agreement >= report.agreement_threshold else "FAIL"
    print(f"  Judge-Human Agreement:  {report.judge_human_agreement:>6.1%}  (threshold: >={report.agreement_threshold:.0%}) [{agree_status}]")
    print()
    
    # Hallucinated samples
    if report.hallucinated_samples:
        print("-" * 60)
        print(f"HALLUCINATED SAMPLES ({len(report.hallucinated_samples)})")
        print("-" * 60)
        for i, sample in enumerate(report.hallucinated_samples, 1):
            print(f"\n  {i}. [{sample['id']}] {sample['contract_type']}")
            print(f"     Question: {sample['question'][:60]}...")
            print(f"     Faithfulness: {sample['faithfulness_score']:.1%}")
            if sample['claims']:
                print(f"     Hallucinated claims:")
                for claim in sample['claims'][:3]:  # Show first 3
                    print(f"       - {claim[:80]}...")
        print()
    
    # Low accuracy samples
    if report.low_accuracy_samples:
        print("-" * 60)
        print(f"LOW ACCURACY SAMPLES ({len(report.low_accuracy_samples)})")
        print("-" * 60)
        for sample in report.low_accuracy_samples:
            print(f"  [{sample['id']}] Accuracy: {sample['accuracy_score']:.1%}")
            print(f"    Question: {sample['question'][:60]}...")
        print()
    
    # Disagreements
    if report.disagreement_samples:
        print("-" * 60)
        print(f"JUDGE-HUMAN DISAGREEMENTS ({len(report.disagreement_samples)})")
        print("-" * 60)
        for sample in report.disagreement_samples:
            judge_says = "hallucination" if sample['judge_detected_hallucination'] else "faithful"
            print(f"  [{sample['id']}] Human: {sample['human_label']}, Judge: {judge_says}")
            print(f"    Question: {sample['question'][:60]}...")
        print()
    
    # Overall status
    print("=" * 60)
    if report.passes_all_thresholds:
        print("STATUS: READY FOR LAUNCH")
    else:
        print("STATUS: NOT READY - Issues need to be addressed")
        
        if report.hallucination_rate > report.hallucination_threshold:
            if report.hallucination_rate > 0.10:
                print("  - CRITICAL: Hallucination rate > 10% - requires immediate fix")
            else:
                print("  - WARNING: Hallucination rate > 5% - investigate and improve")
        
        if report.avg_accuracy < report.accuracy_threshold:
            print("  - WARNING: Accuracy below threshold - review prompt and retrieval")
        
        if report.judge_human_agreement < report.agreement_threshold:
            print("  - WARNING: Judge-Human agreement low - recalibrate evaluators")
    
    print("=" * 60)
    print()


def export_report_json(report: EvaluationReport, filepath: Path) -> None:
    """Export report to JSON file."""
    import json
    from dataclasses import asdict
    
    with open(filepath, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Report exported to {filepath}")


def main():
    """Run report generation on existing results."""
    from src.pipeline import run_quick_evaluation
    
    print("Running quick evaluation to generate report...")
    results, metrics = run_quick_evaluation(max_samples=3)
    
    report = generate_report(results, metrics)
    print_report(report)
    
    # Optionally export
    export_path = Path(__file__).parent.parent / "reports"
    export_path.mkdir(exist_ok=True)
    export_report_json(report, export_path / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")


if __name__ == "__main__":
    main()

