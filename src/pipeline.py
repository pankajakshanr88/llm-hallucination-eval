"""Main evaluation pipeline for hallucination tracking.

This module orchestrates the evaluation process:
1. Load golden dataset
2. Run assistant on test cases
3. Apply evaluators
4. Calculate metrics
5. Track results in LangSmith
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
import uuid

from langsmith import Client
from langsmith.evaluation import evaluate

from .config import settings
from .dataset import GoldenDataset, DatasetEntry, load_golden_dataset
from .evaluators import (
    FaithfulnessEvaluator,
    HallucinationDetector,
    AccuracyScorer,
)
from .mock_assistant import MockDocumentAssistant, ResponseMode


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from an evaluation run."""
    
    # Core metrics
    total_samples: int = 0
    hallucination_count: int = 0
    hallucination_rate: float = 0.0
    
    # Score averages
    avg_faithfulness_score: float = 0.0
    avg_hallucination_score: float = 0.0
    avg_accuracy_score: float = 0.0
    
    # Threshold checks
    passes_hallucination_threshold: bool = False
    passes_accuracy_threshold: bool = False
    
    # Judge-Human agreement
    judge_human_agreement: float = 0.0
    
    # Breakdown by category
    metrics_by_category: dict = field(default_factory=dict)
    
    # Breakdown by severity
    severity_counts: dict = field(default_factory=lambda: {
        "none": 0,
        "minor": 0,
        "major": 0,
        "critical": 0,
    })
    
    def check_launch_readiness(self) -> dict:
        """Check if metrics meet launch requirements from spec."""
        return {
            "hallucination_rate_ok": self.hallucination_rate <= settings.hallucination_rate_threshold,
            "accuracy_ok": self.avg_accuracy_score >= settings.accuracy_threshold,
            "judge_agreement_ok": self.judge_human_agreement >= settings.judge_agreement_threshold,
            "ready_for_launch": (
                self.hallucination_rate <= settings.hallucination_rate_threshold
                and self.avg_accuracy_score >= settings.accuracy_threshold
            ),
            "thresholds": {
                "hallucination_rate": settings.hallucination_rate_threshold,
                "accuracy": settings.accuracy_threshold,
                "judge_agreement": settings.judge_agreement_threshold,
            },
        }


@dataclass
class EvaluationResult:
    """Result for a single evaluation."""
    entry_id: str
    question: str
    contract_type: str
    
    # Assistant response
    assistant_response: str
    
    # Human labels (from golden dataset)
    human_label: str
    human_hallucinated_claims: list[str]
    
    # Judge evaluations
    faithfulness_score: float
    faithfulness_label: str
    hallucination_score: float
    hallucination_detected: bool
    hallucinated_claims_detected: list[str]
    accuracy_score: float
    
    # Agreement
    judge_agrees_with_human: bool


class EvaluationPipeline:
    """Pipeline for running hallucination evaluations."""
    
    def __init__(
        self,
        assistant: Optional[Callable] = None,
        use_langsmith: bool = True,
        project_name: Optional[str] = None,
    ):
        """Initialize the evaluation pipeline.
        
        Args:
            assistant: The assistant to evaluate (callable with LangSmith interface)
            use_langsmith: Whether to track results in LangSmith
            project_name: LangSmith project name
        """
        self.assistant = assistant or MockDocumentAssistant(mode=ResponseMode.LLM)
        self.use_langsmith = use_langsmith
        self.project_name = project_name or settings.langchain_project
        
        # Initialize evaluators
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.hallucination_detector = HallucinationDetector()
        self.accuracy_scorer = AccuracyScorer()
        
        # LangSmith client
        if self.use_langsmith and settings.langchain_api_key:
            self.client = Client()
        else:
            self.client = None
    
    def create_langsmith_dataset(
        self,
        dataset: GoldenDataset,
        dataset_name: Optional[str] = None,
    ) -> str:
        """Create or update a LangSmith dataset from golden dataset.
        
        Returns:
            Dataset ID
        """
        if not self.client:
            raise ValueError("LangSmith client not initialized. Set LANGCHAIN_API_KEY.")
        
        name = dataset_name or f"document-assistant-golden-{datetime.now().strftime('%Y%m%d')}"
        
        # Check if dataset exists
        try:
            existing = self.client.read_dataset(dataset_name=name)
            dataset_id = existing.id
        except Exception:
            # Create new dataset
            new_dataset = self.client.create_dataset(
                dataset_name=name,
                description="Golden dataset for Document Assistant hallucination evaluation",
            )
            dataset_id = new_dataset.id
        
        # Add examples
        for entry in dataset:
            self.client.create_example(
                inputs={
                    "question": entry.question,
                    "contract_text": entry.contract_text,
                    "contract_type": entry.contract_type,
                },
                outputs={
                    "expected_label": entry.human_label.value,
                    "hallucinated_claims": entry.hallucinated_claims,
                },
                metadata={
                    "id": entry.id,
                    "question_category": entry.question_category.value,
                    "reviewer": entry.reviewer_name,
                },
                dataset_id=dataset_id,
            )
        
        return str(dataset_id)
    
    def evaluate_single(self, entry: DatasetEntry) -> EvaluationResult:
        """Evaluate a single dataset entry."""
        # Get assistant response
        inputs = {
            "question": entry.question,
            "contract_text": entry.contract_text,
            "contract_type": entry.contract_type,
        }
        
        assistant_output = self.assistant(inputs)
        response = assistant_output.get("response", "")
        
        # Run evaluators
        faithfulness = self.faithfulness_evaluator.evaluate(
            question=entry.question,
            response=response,
            contract_text=entry.contract_text,
        )
        
        hallucination = self.hallucination_detector.detect(
            question=entry.question,
            response=response,
            contract_text=entry.contract_text,
        )
        
        accuracy = self.accuracy_scorer.score(
            question=entry.question,
            response=response,
            contract_text=entry.contract_text,
        )
        
        # Determine judge-human agreement
        human_says_hallucinated = entry.is_hallucinated()
        judge_says_hallucinated = hallucination.contains_hallucination
        agrees = human_says_hallucinated == judge_says_hallucinated
        
        return EvaluationResult(
            entry_id=entry.id,
            question=entry.question,
            contract_type=entry.contract_type,
            assistant_response=response,
            human_label=entry.human_label.value,
            human_hallucinated_claims=entry.hallucinated_claims,
            faithfulness_score=faithfulness.score,
            faithfulness_label=faithfulness.label,
            hallucination_score=hallucination.score,
            hallucination_detected=hallucination.contains_hallucination,
            hallucinated_claims_detected=[c.claim for c in hallucination.hallucinated_claims],
            accuracy_score=accuracy.score,
            judge_agrees_with_human=agrees,
        )
    
    def run_evaluation(
        self,
        dataset: Optional[GoldenDataset] = None,
        max_samples: Optional[int] = None,
    ) -> tuple[list[EvaluationResult], EvaluationMetrics]:
        """Run full evaluation on the golden dataset.
        
        Args:
            dataset: Golden dataset to use (loads default if not provided)
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Tuple of (list of results, aggregated metrics)
        """
        if dataset is None:
            dataset = load_golden_dataset()
        
        entries = list(dataset)
        if max_samples:
            entries = entries[:max_samples]
        
        results = []
        for entry in entries:
            result = self.evaluate_single(entry)
            results.append(result)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return results, metrics
    
    def run_langsmith_evaluation(
        self,
        dataset_name: str,
        experiment_prefix: Optional[str] = None,
    ) -> dict:
        """Run evaluation using LangSmith's evaluate function.
        
        This provides full LangSmith tracking and dashboard integration.
        """
        if not self.client:
            raise ValueError("LangSmith client not initialized. Set LANGCHAIN_API_KEY.")
        
        prefix = experiment_prefix or f"hallucination-eval-{datetime.now().strftime('%Y%m%d-%H%M')}"
        
        results = evaluate(
            self.assistant,
            data=dataset_name,
            evaluators=[
                self.faithfulness_evaluator,
                self.hallucination_detector,
                self.accuracy_scorer,
            ],
            experiment_prefix=prefix,
            client=self.client,
        )
        
        return results
    
    def _calculate_metrics(self, results: list[EvaluationResult]) -> EvaluationMetrics:
        """Calculate aggregated metrics from evaluation results."""
        if not results:
            return EvaluationMetrics()
        
        total = len(results)
        hallucination_count = sum(1 for r in results if r.hallucination_detected)
        agreement_count = sum(1 for r in results if r.judge_agrees_with_human)
        
        # Calculate averages
        avg_faithfulness = sum(r.faithfulness_score for r in results) / total
        avg_hallucination = sum(r.hallucination_score for r in results) / total
        avg_accuracy = sum(r.accuracy_score for r in results) / total
        
        metrics = EvaluationMetrics(
            total_samples=total,
            hallucination_count=hallucination_count,
            hallucination_rate=hallucination_count / total,
            avg_faithfulness_score=avg_faithfulness,
            avg_hallucination_score=avg_hallucination,
            avg_accuracy_score=avg_accuracy,
            passes_hallucination_threshold=(hallucination_count / total) <= settings.hallucination_rate_threshold,
            passes_accuracy_threshold=avg_accuracy >= settings.accuracy_threshold,
            judge_human_agreement=agreement_count / total,
        )
        
        # Calculate metrics by category
        categories = {}
        for r in results:
            # Find the entry to get category (simplified - in real impl would track this)
            cat = "general"  # Default
            if cat not in categories:
                categories[cat] = {"count": 0, "hallucinations": 0, "accuracy_sum": 0}
            categories[cat]["count"] += 1
            if r.hallucination_detected:
                categories[cat]["hallucinations"] += 1
            categories[cat]["accuracy_sum"] += r.accuracy_score
        
        for cat, data in categories.items():
            if data["count"] > 0:
                data["hallucination_rate"] = data["hallucinations"] / data["count"]
                data["avg_accuracy"] = data["accuracy_sum"] / data["count"]
        
        metrics.metrics_by_category = categories
        
        return metrics


def run_quick_evaluation(max_samples: int = 5) -> tuple[list[EvaluationResult], EvaluationMetrics]:
    """Run a quick evaluation for testing.
    
    Args:
        max_samples: Number of samples to evaluate
        
    Returns:
        Tuple of (results, metrics)
    """
    pipeline = EvaluationPipeline(use_langsmith=False)
    return pipeline.run_evaluation(max_samples=max_samples)

