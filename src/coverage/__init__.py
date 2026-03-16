"""Coverage detection and gap analysis for the evaluation system."""

from .detector import CoverageDetector, CoverageResult, CoverageStatus
from .candidate_queue import CandidateQueue, CandidateEntry
from .reports import CoverageReport, generate_coverage_report

__all__ = [
    "CoverageDetector",
    "CoverageResult", 
    "CoverageStatus",
    "CandidateQueue",
    "CandidateEntry",
    "CoverageReport",
    "generate_coverage_report",
]

