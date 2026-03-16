#!/usr/bin/env python
"""Run evaluation with full LangSmith dashboard integration.

This script:
1. Creates a dataset in LangSmith
2. Runs an experiment using LangSmith's evaluate() function
3. Results appear in the LangSmith dashboard with aggregated metrics

Usage:
    python scripts/run_langsmith_evaluation.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import Client
from langsmith.evaluation import evaluate

from src.config import settings
from src.dataset import load_golden_dataset
from src.mock_assistant import MockDocumentAssistant, ResponseMode
from src.evaluators import (
    FaithfulnessEvaluator,
    HallucinationDetector, 
    AccuracyScorer,
)


def create_or_get_dataset(client: Client, dataset_name: str) -> str:
    """Create a LangSmith dataset or get existing one."""
    
    # Check if dataset exists
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        print(f"  Using existing dataset: {dataset_name}")
        return datasets[0].id
    
    # Load golden dataset
    golden = load_golden_dataset()
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Golden dataset for Document Assistant hallucination evaluation",
    )
    
    print(f"  Created new dataset: {dataset_name}")
    
    # Add examples
    for entry in golden:
        client.create_example(
            inputs={
                "question": entry.question,
                "contract_text": entry.contract_text,
                "contract_type": entry.contract_type,
            },
            outputs={
                "reference_response": entry.assistant_response,
                "human_label": entry.human_label.value,
                "hallucinated_claims": entry.hallucinated_claims,
            },
            metadata={
                "entry_id": entry.id,
                "question_category": entry.question_category.value,
                "reviewer": entry.reviewer_name,
            },
            dataset_id=dataset.id,
        )
    
    print(f"  Added {len(golden)} examples to dataset")
    
    return dataset.id


def main():
    print("=" * 60)
    print("LangSmith Evaluation Pipeline")
    print("=" * 60)
    print()
    
    # Validate configuration
    errors = settings.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    print("Configuration OK")
    print(f"  Project: {settings.langchain_project}")
    print(f"  Model: {settings.eval_model}")
    print()
    
    # Initialize client
    client = Client()
    
    # Create or get dataset
    dataset_name = "document-assistant-golden-dataset"
    print("Setting up dataset...")
    dataset_id = create_or_get_dataset(client, dataset_name)
    print()
    
    # Create assistant
    print("Initializing mock assistant...")
    assistant = MockDocumentAssistant(mode=ResponseMode.LLM)
    print()
    
    # Create evaluators
    print("Setting up evaluators...")
    faithfulness_eval = FaithfulnessEvaluator()
    hallucination_eval = HallucinationDetector()
    accuracy_eval = AccuracyScorer()
    print("  - Faithfulness Evaluator")
    print("  - Hallucination Detector")
    print("  - Accuracy Scorer")
    print()
    
    # Run evaluation
    experiment_name = f"hallucination-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Running evaluation experiment: {experiment_name}")
    print("-" * 60)
    print()
    
    results = evaluate(
        assistant,
        data=dataset_name,
        evaluators=[
            faithfulness_eval,
            hallucination_eval,
            accuracy_eval,
        ],
        experiment_prefix=experiment_name,
        max_concurrency=2,
    )
    
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print()
    print("View your results in the LangSmith dashboard:")
    print(f"  https://smith.langchain.com/")
    print()
    print("Navigate to:")
    print(f"  1. Projects > {settings.langchain_project}")
    print(f"  2. Datasets > {dataset_name}")
    print(f"  3. Experiments > {experiment_name}")
    print()
    print("The dashboard will show:")
    print("  - Aggregated scores for each evaluator")
    print("  - Individual run details")
    print("  - Comparison charts")
    print()
    
    # Print summary of results
    print("-" * 60)
    print("QUICK SUMMARY")
    print("-" * 60)
    
    # Calculate aggregate metrics from results
    faithfulness_scores = []
    hallucination_scores = []
    accuracy_scores = []
    
    for result in results:
        if hasattr(result, 'evaluation_results'):
            for eval_result in result.evaluation_results.get('results', []):
                if eval_result.key == 'faithfulness':
                    faithfulness_scores.append(eval_result.score or 0)
                elif eval_result.key == 'hallucination':
                    hallucination_scores.append(eval_result.score or 0)
                elif eval_result.key == 'accuracy':
                    accuracy_scores.append(eval_result.score or 0)
    
    if faithfulness_scores:
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        print(f"  Avg Faithfulness Score: {avg_faithfulness:.1%}")
    
    if hallucination_scores:
        avg_hallucination = sum(hallucination_scores) / len(hallucination_scores)
        # Hallucination score is inverted (1 = no hallucination)
        hallucination_rate = 1 - avg_hallucination
        print(f"  Hallucination Rate: {hallucination_rate:.1%}")
    
    if accuracy_scores:
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        print(f"  Avg Accuracy Score: {avg_accuracy:.1%}")
    
    print()
    print("For detailed metrics, check the LangSmith dashboard.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

