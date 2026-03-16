"""Golden dataset structure and loader for hallucination evaluation."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class HumanLabel(str, Enum):
    """Human-assigned labels for response faithfulness."""
    FAITHFUL = "faithful"
    PARTIALLY_FAITHFUL = "partially_faithful"
    UNFAITHFUL = "unfaithful"


class QuestionCategory(str, Enum):
    """Categories of questions as defined in the spec."""
    SUMMARIZATION = "summarization"
    CLAUSE_DEEP_DIVE = "clause_deep_dive"
    NEXT_STEPS = "next_steps"
    LEGAL_CONCEPTS = "legal_concepts"
    FOLLOW_UP = "follow_up"
    GENERAL = "general"


@dataclass
class DatasetEntry:
    """A single entry in the golden dataset."""
    
    # Unique identifier
    id: str
    
    # Contract information
    contract_type: str
    contract_text: str
    
    # Question and response
    question: str
    question_category: QuestionCategory
    assistant_response: str
    
    # Human evaluation
    human_label: HumanLabel
    human_notes: str = ""
    reviewer_name: str = ""
    
    # Detailed grading (from spec)
    accuracy_score: Optional[float] = None      # 0-1 scale
    faithfulness_score: Optional[float] = None  # 0-1 scale
    completeness_score: Optional[float] = None  # 0-1 scale
    tone_score: Optional[float] = None          # 0-1 scale
    clarity_score: Optional[float] = None       # 0-1 scale
    
    # Hallucination details
    hallucinated_claims: list[str] = field(default_factory=list)
    
    def is_hallucinated(self) -> bool:
        """Check if this entry contains any hallucination."""
        return self.human_label in (
            HumanLabel.PARTIALLY_FAITHFUL, 
            HumanLabel.UNFAITHFUL
        )
    
    def to_dict(self) -> dict:
        """Convert entry to dictionary for serialization."""
        return {
            "id": self.id,
            "contract_type": self.contract_type,
            "contract_text": self.contract_text,
            "question": self.question,
            "question_category": self.question_category.value,
            "assistant_response": self.assistant_response,
            "human_label": self.human_label.value,
            "human_notes": self.human_notes,
            "reviewer_name": self.reviewer_name,
            "accuracy_score": self.accuracy_score,
            "faithfulness_score": self.faithfulness_score,
            "completeness_score": self.completeness_score,
            "tone_score": self.tone_score,
            "clarity_score": self.clarity_score,
            "hallucinated_claims": self.hallucinated_claims,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatasetEntry":
        """Create entry from dictionary."""
        return cls(
            id=data["id"],
            contract_type=data["contract_type"],
            contract_text=data["contract_text"],
            question=data["question"],
            question_category=QuestionCategory(data["question_category"]),
            assistant_response=data["assistant_response"],
            human_label=HumanLabel(data["human_label"]),
            human_notes=data.get("human_notes", ""),
            reviewer_name=data.get("reviewer_name", ""),
            accuracy_score=data.get("accuracy_score"),
            faithfulness_score=data.get("faithfulness_score"),
            completeness_score=data.get("completeness_score"),
            tone_score=data.get("tone_score"),
            clarity_score=data.get("clarity_score"),
            hallucinated_claims=data.get("hallucinated_claims", []),
        )


@dataclass
class GoldenDataset:
    """Collection of golden dataset entries for evaluation."""
    
    entries: list[DatasetEntry] = field(default_factory=list)
    
    def add_entry(self, entry: DatasetEntry) -> None:
        """Add an entry to the dataset."""
        self.entries.append(entry)
    
    def get_by_category(self, category: QuestionCategory) -> list[DatasetEntry]:
        """Get all entries for a specific question category."""
        return [e for e in self.entries if e.question_category == category]
    
    def get_by_contract_type(self, contract_type: str) -> list[DatasetEntry]:
        """Get all entries for a specific contract type."""
        return [e for e in self.entries if e.contract_type == contract_type]
    
    def get_hallucinated_entries(self) -> list[DatasetEntry]:
        """Get all entries marked as containing hallucinations."""
        return [e for e in self.entries if e.is_hallucinated()]
    
    def get_faithful_entries(self) -> list[DatasetEntry]:
        """Get all entries marked as faithful."""
        return [e for e in self.entries if not e.is_hallucinated()]
    
    def calculate_human_hallucination_rate(self) -> float:
        """Calculate hallucination rate based on human labels."""
        if not self.entries:
            return 0.0
        hallucinated = len(self.get_hallucinated_entries())
        return hallucinated / len(self.entries)
    
    def to_langsmith_dataset(self) -> list[dict]:
        """Convert to format suitable for LangSmith dataset creation."""
        return [
            {
                "inputs": {
                    "question": e.question,
                    "contract_text": e.contract_text,
                    "contract_type": e.contract_type,
                },
                "outputs": {
                    "response": e.assistant_response,
                    "human_label": e.human_label.value,
                    "hallucinated_claims": e.hallucinated_claims,
                },
                "metadata": {
                    "id": e.id,
                    "question_category": e.question_category.value,
                    "reviewer_name": e.reviewer_name,
                }
            }
            for e in self.entries
        ]
    
    def save(self, filepath: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "version": "1.0",
            "entries": [e.to_dict() for e in self.entries]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "GoldenDataset":
        """Load dataset from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entries = [DatasetEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries=entries)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self):
        return iter(self.entries)


def load_golden_dataset() -> GoldenDataset:
    """Load the golden dataset from the default location."""
    dataset_path = Path(__file__).parent / "sample_data.json"
    if dataset_path.exists():
        return GoldenDataset.load(dataset_path)
    return GoldenDataset()

