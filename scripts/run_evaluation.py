#!/usr/bin/env python
"""Run the full hallucination evaluation pipeline.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --max-samples 5
    python scripts/run_evaluation.py --use-langsmith
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.dataset import load_golden_dataset
from src.pipeline import EvaluationPipeline
from src.mock_assistant import MockDocumentAssistant, ResponseMode
from scripts.generate_report import generate_report, print_report


def main():
    parser = argparse.ArgumentParser(description="Run hallucination evaluation")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--use-langsmith",
        action="store_true",
        help="Track results in LangSmith",
    )
    parser.add_argument(
        "--assistant-mode",
        choices=["faithful", "hallucinating", "mixed", "llm"],
        default="llm",
        help="Mode for mock assistant",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running evaluation",
    )
    args = parser.parse_args()
    
    # Validate configuration
    print("=" * 60)
    print("Hallucination Tracking Evaluation System")
    print("=" * 60)
    print()
    print("Configuration:")
    print(settings)
    print()
    
    errors = settings.validate()
    if errors and not args.validate_only:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print()
        print("Please set the required environment variables in .env file")
        print("See README.md for setup instructions")
        return 1
    
    if args.validate_only:
        if errors:
            print("Configuration has errors (see above)")
            return 1
        print("Configuration is valid!")
        return 0
    
    # Load dataset
    print("Loading golden dataset...")
    dataset = load_golden_dataset()
    print(f"  Loaded {len(dataset)} entries")
    print(f"  Hallucination rate in dataset: {dataset.calculate_human_hallucination_rate():.1%}")
    print()
    
    # Set up assistant
    mode_map = {
        "faithful": ResponseMode.FAITHFUL,
        "hallucinating": ResponseMode.HALLUCINATING,
        "mixed": ResponseMode.MIXED,
        "llm": ResponseMode.LLM,
    }
    assistant_mode = mode_map[args.assistant_mode]
    assistant = MockDocumentAssistant(mode=assistant_mode)
    print(f"Using assistant mode: {assistant_mode.value}")
    print()
    
    # Create pipeline
    pipeline = EvaluationPipeline(
        assistant=assistant,
        use_langsmith=args.use_langsmith,
    )
    
    # Run evaluation
    print("Running evaluation...")
    print("-" * 60)
    
    results, metrics = pipeline.run_evaluation(
        dataset=dataset,
        max_samples=args.max_samples,
    )
    
    print(f"Evaluated {len(results)} samples")
    print()
    
    # Generate and print report
    report = generate_report(results, metrics)
    print_report(report)
    
    # Check launch readiness
    print()
    print("=" * 60)
    print("Launch Readiness Check")
    print("=" * 60)
    readiness = metrics.check_launch_readiness()
    
    print(f"\n  Hallucination Rate: {metrics.hallucination_rate:.1%} (threshold: ≤{readiness['thresholds']['hallucination_rate']:.0%})")
    print(f"    Status: {'PASS' if readiness['hallucination_rate_ok'] else 'FAIL'}")
    
    print(f"\n  Accuracy: {metrics.avg_accuracy_score:.1%} (threshold: ≥{readiness['thresholds']['accuracy']:.0%})")
    print(f"    Status: {'PASS' if readiness['accuracy_ok'] else 'FAIL'}")
    
    print(f"\n  Judge-Human Agreement: {metrics.judge_human_agreement:.1%} (threshold: ≥{readiness['thresholds']['judge_agreement']:.0%})")
    print(f"    Status: {'PASS' if readiness['judge_agreement_ok'] else 'FAIL'}")
    
    print()
    if readiness["ready_for_launch"]:
        print("  RESULT: READY FOR LAUNCH")
    else:
        print("  RESULT: NOT READY - Address issues before launch")
    print()
    
    return 0 if readiness["ready_for_launch"] else 1


if __name__ == "__main__":
    sys.exit(main())

