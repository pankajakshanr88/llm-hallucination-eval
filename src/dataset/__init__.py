"""Golden dataset module for hallucination evaluation."""

from .golden_dataset import (
    GoldenDataset,
    DatasetEntry,
    HumanLabel,
    QuestionCategory,
    load_golden_dataset,
)

__all__ = [
    "GoldenDataset",
    "DatasetEntry",
    "HumanLabel",
    "QuestionCategory",
    "load_golden_dataset",
]

