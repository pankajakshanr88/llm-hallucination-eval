#!/usr/bin/env python
"""
DEMO SCRIPT FOR MANAGEMENT
==========================

This script demonstrates how the Hallucination Tracking System handles
different scenarios, including edge cases. Run this to show stakeholders
what happens in various situations.

Usage:
    python scripts/demo_for_management.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.dataset import load_golden_dataset
from src.coverage import (
    CoverageDetector,
    CoverageStatus,
    CandidateQueue,
    generate_coverage_report,
)
from src.evaluators import GroundednessEvaluator


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str):
    """Print a formatted subheader."""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60 + "\n")


def demo_scenario_1_golden_match():
    """Scenario 1: Query has golden dataset coverage."""
    print_header("SCENARIO 1: Query WITH Golden Dataset Coverage")
    
    print("This is the ideal case - we have a pre-labeled reference answer.\n")
    
    # Example query that matches golden dataset
    query = {
        "question": "What is the monthly rent amount?",
        "contract_type": "Residential Lease Agreement",
        "contract_text": "... (lease contract) ...",
    }
    
    print("📥 INCOMING QUERY:")
    print(f"   Question: '{query['question']}'")
    print(f"   Contract Type: {query['contract_type']}")
    print()
    
    # Check coverage
    detector = CoverageDetector()
    result = detector.detect(
        question=query["question"],
        contract_type=query["contract_type"],
    )
    
    print("🔍 COVERAGE CHECK:")
    print(f"   Status: {result.status.value}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Matched Golden Entry: {result.matched_entry.id if result.matched_entry else 'None'}")
    print(f"   Similarity Score: {result.similarity_score:.0%}")
    print()
    
    print("✅ EVALUATION PATH:")
    print("   → Full evaluation with golden reference")
    print("   → Calculate: Accuracy, Faithfulness, Hallucination")
    print("   → Compare AI response to human-labeled reference")
    print()
    
    print("📊 METRICS INCLUDED IN:")
    print("   ✓ Hallucination Rate (counted)")
    print("   ✓ Accuracy Rate (counted)")
    print("   ✓ Judge-Human Agreement (counted)")


def demo_scenario_2_no_coverage():
    """Scenario 2: Query has NO golden dataset coverage."""
    print_header("SCENARIO 2: Query WITHOUT Golden Dataset Coverage")
    
    print("This query is about a contract type or question we haven't labeled yet.\n")
    
    # Example query without coverage
    query = {
        "question": "What are the intellectual property rights in this contract?",
        "contract_type": "Software License Agreement",  # Not in our golden set
        "contract_text": "SOFTWARE LICENSE AGREEMENT...",
    }
    
    print("📥 INCOMING QUERY:")
    print(f"   Question: '{query['question']}'")
    print(f"   Contract Type: {query['contract_type']}")
    print()
    
    # Check coverage
    detector = CoverageDetector()
    result = detector.detect(
        question=query["question"],
        contract_type=query["contract_type"],
    )
    
    print("🔍 COVERAGE CHECK:")
    print(f"   Status: {result.status.value}")
    print(f"   Contract Type Match: {result.contract_type_match}")
    print(f"   Recommendation: {result.recommendation}")
    print()
    
    print("⚠️  EVALUATION PATH:")
    print("   → Rubric-only evaluation (no reference to compare)")
    print("   → Calculate: Groundedness, Relevance, Clarity, Tone")
    print("   → Cannot calculate hallucination without reference")
    print()
    
    print("📊 METRICS:")
    print("   ✗ NOT included in Hallucination Rate")
    print("   ✗ NOT included in Accuracy Rate")
    print("   ✓ Included in Groundedness metrics (separate category)")
    print()
    
    print("📝 ACTION TAKEN:")
    print("   → Logged to Candidate Queue for Legal review")
    print("   → If pattern appears ≥5 times → flagged for review")
    print("   → If pattern appears ≥10 times → HIGH PRIORITY")


def demo_scenario_3_out_of_scope():
    """Scenario 3: Query is asking for legal advice (out of scope)."""
    print_header("SCENARIO 3: Out-of-Scope Query (Legal Advice)")
    
    print("This query is asking for legal advice, which we don't provide.\n")
    
    # Example out-of-scope query
    query = {
        "question": "Should I sign this contract? Is it a good deal?",
        "contract_type": "Residential Lease Agreement",
        "contract_text": "... (lease contract) ...",
    }
    
    print("📥 INCOMING QUERY:")
    print(f"   Question: '{query['question']}'")
    print(f"   Contract Type: {query['contract_type']}")
    print()
    
    # Check coverage
    detector = CoverageDetector()
    result = detector.detect(
        question=query["question"],
        contract_type=query["contract_type"],
    )
    
    print("🔍 COVERAGE CHECK:")
    print(f"   Status: {result.status.value}")
    print(f"   Recommendation: {result.recommendation}")
    print()
    
    print("🚫 EVALUATION PATH:")
    print("   → Route to guardrail response")
    print("   → AI should respond: 'I can't provide legal advice...'")
    print("   → Track as 'Good Abstention' if handled correctly")
    print()
    
    print("📊 METRICS:")
    print("   ✗ EXCLUDED from Hallucination Rate")
    print("   ✗ EXCLUDED from Accuracy Rate")
    print("   ✓ Tracked in 'Guardrail Compliance' metric")
    print("   ✓ Tracked in 'Good Abstention Rate'")


def demo_scenario_4_partial_match():
    """Scenario 4: Query partially matches golden dataset."""
    print_header("SCENARIO 4: Partial Match (Same Contract, Different Question)")
    
    print("We have coverage for this contract type, but not this exact question.\n")
    
    # Example partial match
    query = {
        "question": "Can I sublease the apartment to someone else?",
        "contract_type": "Residential Lease Agreement",
        "contract_text": "... (lease contract) ...",
    }
    
    print("📥 INCOMING QUERY:")
    print(f"   Question: '{query['question']}'")
    print(f"   Contract Type: {query['contract_type']}")
    print()
    
    # Check coverage
    detector = CoverageDetector()
    result = detector.detect(
        question=query["question"],
        contract_type=query["contract_type"],
    )
    
    print("🔍 COVERAGE CHECK:")
    print(f"   Status: {result.status.value}")
    print(f"   Contract Type Match: {result.contract_type_match}")
    print(f"   Similarity to Nearest: {result.similarity_score:.0%}")
    print(f"   Nearest Golden Entry: {result.matched_entry.id if result.matched_entry else 'None'}")
    print()
    
    print("⚡ EVALUATION PATH:")
    print("   → Hybrid evaluation")
    print("   → Groundedness check (rubric-only)")
    print("   → Consider adding to golden set if frequent")
    print()
    
    print("📊 METRICS:")
    print("   ✗ NOT in primary Hallucination Rate")
    print("   ✓ Included in Groundedness metrics")
    print("   ✓ Flagged for potential golden set addition")


def demo_candidate_queue():
    """Demonstrate the candidate queue system."""
    print_header("CANDIDATE QUEUE SYSTEM")
    
    print("Queries without coverage are logged for Legal team review.\n")
    
    # Create sample queue
    queue = CandidateQueue()
    
    # Simulate adding candidates
    sample_candidates = [
        ("What is the warranty period?", "Bill of Sale", 7),
        ("Can I terminate early without penalty?", "Service Agreement", 12),
        ("What happens if payment is late?", "Loan Agreement", 3),
        ("Who is responsible for repairs?", "Residential Lease Agreement", 5),
        ("What are my obligations under this NDA?", "Non-Disclosure Agreement", 8),
    ]
    
    for question, contract_type, count in sample_candidates:
        for _ in range(count):
            queue.add_candidate(
                question=question,
                contract_type=contract_type,
                contract_text="[Contract text...]",
                assistant_response="[AI response...]",
            )
    
    print("📋 CURRENT QUEUE STATUS:")
    stats = queue.get_stats()
    print(f"   Total Candidates: {stats['total_candidates']}")
    print(f"   Pending Review: {stats['pending']}")
    print(f"   Ready for Review (≥5 occurrences): {stats['ready_for_review']}")
    print(f"   High Priority (≥10 occurrences): {stats['high_priority']}")
    print()
    
    print("📑 CANDIDATES READY FOR LEGAL REVIEW:")
    for entry in queue.get_pending_for_review():
        priority_icon = "🔴" if entry.priority.value == "high" else "🟡"
        print(f"   {priority_icon} [{entry.occurrence_count}x] {entry.contract_type}")
        print(f"      '{entry.question}'")
    print()
    
    print("⚙️  WORKFLOW:")
    print("   1. System logs new question patterns automatically")
    print("   2. Patterns appearing ≥5 times → Ready for Review")
    print("   3. Patterns appearing ≥10 times → High Priority")
    print("   4. Legal reviews weekly batch of top candidates")
    print("   5. Approved candidates → Added to Golden Dataset")


def demo_coverage_report():
    """Demonstrate the coverage report for management."""
    print_header("COVERAGE REPORT FOR MANAGEMENT")
    
    print("This is what stakeholders see in the weekly report.\n")
    
    # Simulate evaluation results
    simulated_queries = [
        {"question": "What is the rent?", "contract_type": "Residential Lease Agreement"},
        {"question": "Can I have pets?", "contract_type": "Residential Lease Agreement"},
        {"question": "What powers does the agent have?", "contract_type": "Power of Attorney"},
        {"question": "How long does this NDA last?", "contract_type": "Non-Disclosure Agreement"},
        {"question": "What is my salary?", "contract_type": "Employment Contract"},
        # Uncovered queries
        {"question": "What is the warranty?", "contract_type": "Software License"},
        {"question": "Can I cancel anytime?", "contract_type": "Subscription Agreement"},
        # Out of scope
        {"question": "Should I sign this?", "contract_type": "Residential Lease Agreement"},
        {"question": "Is this enforceable?", "contract_type": "Employment Contract"},
        # More covered
        {"question": "What is the monthly rent amount?", "contract_type": "Residential Lease Agreement"},
    ]
    
    detector = CoverageDetector()
    queue = CandidateQueue()
    
    report = generate_coverage_report(
        queries_evaluated=simulated_queries,
        coverage_detector=detector,
        candidate_queue=queue,
        report_period="Demo Evaluation",
    )
    
    print("=" * 50)
    print("  HALLUCINATION TRACKING - WEEKLY REPORT")
    print("=" * 50)
    print(f"  Generated: {report.generated_at[:10]}")
    print(f"  Period: {report.report_period}")
    print("=" * 50)
    print()
    
    print("📊 QUERY DISTRIBUTION")
    print("─" * 40)
    total = report.total_queries_evaluated
    print(f"  Total Queries Evaluated: {total}")
    print()
    print(f"  ✅ Golden Match:    {report.golden_matched_count:3d}  ({report.golden_coverage_rate:5.1%})")
    print(f"  🟡 Partial Match:   {report.partial_matched_count:3d}  ({report.partial_coverage_rate:5.1%})")
    print(f"  ❌ No Coverage:     {report.no_coverage_count:3d}  ({report.no_coverage_rate:5.1%})")
    print(f"  🚫 Out of Scope:    {report.out_of_scope_count:3d}  ({report.out_of_scope_rate:5.1%})")
    print()
    
    print("📈 QUALITY METRICS (Golden-Matched Only)")
    print("─" * 40)
    print(f"  Hallucination Rate:  {report.hallucination_rate_golden:5.1%}  (target: ≤5%)")
    print(f"  Accuracy:            {report.accuracy_golden:5.1%}  (target: ≥85%)")
    print(f"  Faithfulness:        {report.faithfulness_golden:5.1%}  (target: ≥85%)")
    print()
    print("  ⚠️  Note: These metrics are calculated ONLY on queries")
    print("      with golden dataset coverage.")
    print()
    
    print("📈 RUBRIC METRICS (Non-Golden)")
    print("─" * 40)
    print(f"  Groundedness:        {report.groundedness_non_golden:5.1%}")
    print()
    print("  ℹ️  Groundedness is measured for queries without golden")
    print("      coverage using rubric-only evaluation.")
    print()
    
    print("📝 CANDIDATE QUEUE")
    print("─" * 40)
    print(f"  Pending Review:      {report.candidates_pending_review}")
    print(f"  High Priority:       {report.candidates_high_priority}")
    print()
    
    print("🎯 RECOMMENDATIONS")
    print("─" * 40)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    if not report.recommendations:
        print("  ✓ No immediate actions required")
    print()
    
    print("🚀 LAUNCH READINESS")
    print("─" * 40)
    if report.is_launch_ready():
        print("  ✅ READY FOR LAUNCH")
        print("     All metrics meet thresholds")
    else:
        print("  ❌ NOT READY")
        print("     Address recommendations before launch")


def demo_how_many_is_enough():
    """Demonstrate golden dataset sufficiency criteria."""
    print_header("HOW MANY GOLDEN QUESTIONS IS ENOUGH?")
    
    print("We use metric-based criteria, not arbitrary counts.\n")
    
    print("📏 SUFFICIENCY CRITERIA PER CONTRACT TYPE:")
    print("─" * 50)
    print()
    print("  A contract type is LAUNCH-READY when:")
    print()
    print("  1. COVERAGE")
    print("     └─ ≥15 Legal-authored golden questions")
    print("     └─ ≥50 total evaluated queries (golden + synthetic)")
    print("     └─ All 5 major intent categories covered:")
    print("        • Obligations")
    print("        • Termination/Cancellation")
    print("        • Payment Terms")
    print("        • Liability")
    print("        • Duration/Renewal")
    print()
    print("  2. QUALITY METRICS")
    print("     └─ Hallucination Rate < 5%")
    print("     └─ Accuracy ≥ 85%")
    print("     └─ Faithfulness ≥ 90%")
    print()
    print("  3. JUDGE CALIBRATION")
    print("     └─ LLM-Judge agrees with human labels ≥ 80%")
    print()
    
    # Show current coverage
    dataset = load_golden_dataset()
    detector = CoverageDetector()
    stats = detector.get_coverage_stats()
    
    print("📊 CURRENT GOLDEN DATASET STATUS:")
    print("─" * 50)
    print(f"  Total Entries: {stats['total_golden_entries']}")
    print(f"  Contract Types: {stats['contract_types_covered']}")
    print()
    
    for ct, ct_stats in stats["contract_types"].items():
        status = "✅" if ct_stats["total_questions"] >= 15 else "⚠️"
        print(f"  {status} {ct.title()}")
        print(f"     Questions: {ct_stats['total_questions']}/15 minimum")
        print(f"     Categories: {ct_stats['categories']}")
        print()
    
    print("📈 AUGMENTATION STRATEGY:")
    print("─" * 50)
    print()
    print("  Phase 1: Legal authors seed set (10-20 per contract)")
    print("           ↓")
    print("  Phase 2: GPT generates variations (50-100 per contract)")
    print("           ↓")
    print("  Phase 3: Legal spot-checks 10-20% of generated")
    print("           ↓")
    print("  Phase 4: Run evaluation to verify metrics")
    print()


def main():
    """Run all demos for management presentation."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", help="Pause between scenarios")
    args = parser.parse_args()
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  HALLUCINATION TRACKING SYSTEM - MANAGEMENT DEMO".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\nThis demo shows how the system handles different scenarios.")
    print("Each scenario demonstrates a specific edge case and how we address it.")
    
    def pause():
        if args.interactive:
            input("\n[Press Enter to continue to next scenario...]")
        else:
            print("\n" + "·" * 50 + "\n")
    
    # Run demos
    demo_scenario_1_golden_match()
    pause()
    
    demo_scenario_2_no_coverage()
    pause()
    
    demo_scenario_3_out_of_scope()
    pause()
    
    demo_scenario_4_partial_match()
    pause()
    
    demo_candidate_queue()
    pause()
    
    demo_coverage_report()
    pause()
    
    demo_how_many_is_enough()
    
    print_header("DEMO COMPLETE")
    print("Key Takeaways:")
    print()
    print("  1. ✅ Golden-matched queries → Full evaluation with reference")
    print("  2. 🟡 Partial/No coverage → Rubric-only evaluation")
    print("  3. 🚫 Out-of-scope → Guardrail response, excluded from metrics")
    print("  4. 📝 Gaps logged → Candidate queue for Legal review")
    print("  5. 📊 Metrics split → Clear reporting of what % represents")
    print()
    print("Questions? Let's discuss!")
    print()


if __name__ == "__main__":
    main()

