"""Unit tests for LLM-as-Judge evaluators.

These tests verify evaluator behavior without making actual LLM calls.
For integration tests that use real LLMs, see test_integration.py.
"""

import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    DatasetEntry,
    GoldenDataset,
    HumanLabel,
    QuestionCategory,
    load_golden_dataset,
)
from src.evaluators.faithfulness import FaithfulnessResult
from src.evaluators.hallucination import HallucinationResult, HallucinatedClaim
from src.evaluators.accuracy import AccuracyResult


class TestDatasetStructure:
    """Tests for dataset loading and structure."""
    
    def test_load_golden_dataset(self):
        """Test that golden dataset loads correctly."""
        dataset = load_golden_dataset()
        assert len(dataset) > 0
        assert isinstance(dataset, GoldenDataset)
    
    def test_dataset_entry_structure(self):
        """Test DatasetEntry has required fields."""
        dataset = load_golden_dataset()
        entry = dataset.entries[0]
        
        assert entry.id is not None
        assert entry.contract_type is not None
        assert entry.contract_text is not None
        assert entry.question is not None
        assert entry.assistant_response is not None
        assert isinstance(entry.human_label, HumanLabel)
        assert isinstance(entry.question_category, QuestionCategory)
    
    def test_dataset_has_hallucinated_entries(self):
        """Test that dataset contains both faithful and hallucinated entries."""
        dataset = load_golden_dataset()
        
        faithful = dataset.get_faithful_entries()
        hallucinated = dataset.get_hallucinated_entries()
        
        assert len(faithful) > 0, "Dataset should have faithful entries"
        assert len(hallucinated) > 0, "Dataset should have hallucinated entries"
    
    def test_hallucination_rate_calculation(self):
        """Test hallucination rate calculation."""
        dataset = load_golden_dataset()
        rate = dataset.calculate_human_hallucination_rate()
        
        assert 0 <= rate <= 1
        assert rate > 0  # We have some hallucinated entries
    
    def test_langsmith_format_conversion(self):
        """Test conversion to LangSmith dataset format."""
        dataset = load_golden_dataset()
        langsmith_data = dataset.to_langsmith_dataset()
        
        assert len(langsmith_data) == len(dataset)
        
        for item in langsmith_data:
            assert "inputs" in item
            assert "outputs" in item
            assert "metadata" in item
            assert "question" in item["inputs"]
            assert "contract_text" in item["inputs"]


class TestFaithfulnessResult:
    """Tests for FaithfulnessResult structure."""
    
    def test_faithful_result(self):
        """Test faithful result structure."""
        result = FaithfulnessResult(
            label="faithful",
            score=1.0,
            reasoning="All claims supported by contract",
            unsupported_claims=[],
        )
        
        assert result.label == "faithful"
        assert result.score == 1.0
        assert len(result.unsupported_claims) == 0
    
    def test_unfaithful_result(self):
        """Test unfaithful result structure."""
        result = FaithfulnessResult(
            label="unfaithful",
            score=0.2,
            reasoning="Response contains invented clauses",
            unsupported_claims=["The contract requires notarization"],
        )
        
        assert result.label == "unfaithful"
        assert result.score < 0.5
        assert len(result.unsupported_claims) > 0
    
    def test_to_evaluation_result(self):
        """Test conversion to LangSmith EvaluationResult."""
        result = FaithfulnessResult(
            label="faithful",
            score=0.9,
            reasoning="Test",
            unsupported_claims=[],
        )
        
        eval_result = result.to_evaluation_result()
        
        assert eval_result.key == "faithfulness"
        assert eval_result.score == 0.9


class TestHallucinationResult:
    """Tests for HallucinationResult structure."""
    
    def test_no_hallucination_result(self):
        """Test result with no hallucination."""
        result = HallucinationResult(
            contains_hallucination=False,
            hallucination_count=0,
            severity="none",
            hallucinated_claims=[],
            score=1.0,
            reasoning="Response is grounded in contract",
        )
        
        assert not result.contains_hallucination
        assert result.score == 1.0
        assert result.severity == "none"
    
    def test_hallucination_detected_result(self):
        """Test result with detected hallucination."""
        claims = [
            HallucinatedClaim(
                claim="The contract has a 90-day cancellation period",
                reason="Contract states 60 days, not 90",
                severity="major",
            )
        ]
        
        result = HallucinationResult(
            contains_hallucination=True,
            hallucination_count=1,
            severity="major",
            hallucinated_claims=claims,
            score=0.3,
            reasoning="Found one major hallucination",
        )
        
        assert result.contains_hallucination
        assert result.hallucination_count == 1
        assert len(result.hallucinated_claims) == 1
        assert result.hallucinated_claims[0].severity == "major"
    
    def test_to_evaluation_result(self):
        """Test conversion to LangSmith EvaluationResult."""
        result = HallucinationResult(
            contains_hallucination=False,
            hallucination_count=0,
            severity="none",
            score=1.0,
            reasoning="Test",
        )
        
        eval_result = result.to_evaluation_result()
        
        assert eval_result.key == "hallucination"
        assert eval_result.score == 1.0
        assert "contains_hallucination" in eval_result.extra


class TestAccuracyResult:
    """Tests for AccuracyResult structure."""
    
    def test_high_accuracy_result(self):
        """Test high accuracy result."""
        result = AccuracyResult(
            score=0.95,
            correctness=0.95,
            completeness=0.90,
            relevance=1.0,
            precision=0.95,
            reasoning="Excellent response",
            errors=[],
        )
        
        assert result.score >= 0.9
        assert len(result.errors) == 0
    
    def test_low_accuracy_result(self):
        """Test low accuracy result."""
        result = AccuracyResult(
            score=0.4,
            correctness=0.3,
            completeness=0.5,
            relevance=0.6,
            precision=0.3,
            reasoning="Multiple errors found",
            errors=["Incorrect date", "Wrong party name"],
        )
        
        assert result.score < 0.5
        assert len(result.errors) > 0
    
    def test_to_evaluation_result(self):
        """Test conversion to LangSmith EvaluationResult."""
        result = AccuracyResult(
            score=0.8,
            correctness=0.8,
            completeness=0.8,
            relevance=0.8,
            precision=0.8,
            reasoning="Test",
            errors=[],
        )
        
        eval_result = result.to_evaluation_result()
        
        assert eval_result.key == "accuracy"
        assert eval_result.score == 0.8


class TestMetricsCalculation:
    """Tests for metrics calculation."""
    
    def test_threshold_checks(self):
        """Test that threshold checks work correctly."""
        from src.pipeline import EvaluationMetrics
        from src.config import settings
        
        # Passing metrics
        passing = EvaluationMetrics(
            total_samples=100,
            hallucination_rate=0.03,
            avg_accuracy_score=0.90,
            judge_human_agreement=0.85,
        )
        
        readiness = passing.check_launch_readiness()
        assert readiness["hallucination_rate_ok"]
        assert readiness["accuracy_ok"]
        
        # Failing metrics
        failing = EvaluationMetrics(
            total_samples=100,
            hallucination_rate=0.15,  # Above threshold
            avg_accuracy_score=0.70,  # Below threshold
            judge_human_agreement=0.60,
        )
        
        readiness = failing.check_launch_readiness()
        assert not readiness["hallucination_rate_ok"]
        assert not readiness["accuracy_ok"]
        assert not readiness["ready_for_launch"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

