"""Candidate queue for queries that need Legal review."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class CandidateStatus(str, Enum):
    """Status of a candidate entry."""
    PENDING = "pending"              # Awaiting Legal review
    APPROVED = "approved"            # Approved for golden dataset
    REJECTED = "rejected"            # Not suitable for golden dataset
    DEFERRED = "deferred"            # Review later


class CandidatePriority(str, Enum):
    """Priority level for candidates."""
    HIGH = "high"                    # Frequent pattern, needs review soon
    MEDIUM = "medium"                # Moderate frequency
    LOW = "low"                      # Rare pattern


@dataclass
class CandidateEntry:
    """A candidate query for potential addition to golden dataset."""
    
    id: str
    question: str
    contract_type: str
    contract_text: str
    assistant_response: str
    
    # Metadata
    first_seen: str                  # ISO timestamp
    occurrence_count: int = 1        # How many times this pattern appeared
    last_seen: str = ""
    
    # Classification
    status: CandidateStatus = CandidateStatus.PENDING
    priority: CandidatePriority = CandidatePriority.LOW
    
    # Coverage info
    similarity_to_nearest: float = 0.0
    nearest_golden_id: Optional[str] = None
    
    # Review info
    reviewer: str = ""
    review_notes: str = ""
    reviewed_at: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "contract_type": self.contract_type,
            "contract_text": self.contract_text[:500] + "..." if len(self.contract_text) > 500 else self.contract_text,
            "assistant_response": self.assistant_response,
            "first_seen": self.first_seen,
            "occurrence_count": self.occurrence_count,
            "last_seen": self.last_seen,
            "status": self.status.value,
            "priority": self.priority.value,
            "similarity_to_nearest": self.similarity_to_nearest,
            "nearest_golden_id": self.nearest_golden_id,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "reviewed_at": self.reviewed_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CandidateEntry":
        return cls(
            id=data["id"],
            question=data["question"],
            contract_type=data["contract_type"],
            contract_text=data.get("contract_text", ""),
            assistant_response=data.get("assistant_response", ""),
            first_seen=data["first_seen"],
            occurrence_count=data.get("occurrence_count", 1),
            last_seen=data.get("last_seen", ""),
            status=CandidateStatus(data.get("status", "pending")),
            priority=CandidatePriority(data.get("priority", "low")),
            similarity_to_nearest=data.get("similarity_to_nearest", 0.0),
            nearest_golden_id=data.get("nearest_golden_id"),
            reviewer=data.get("reviewer", ""),
            review_notes=data.get("review_notes", ""),
            reviewed_at=data.get("reviewed_at", ""),
        )


@dataclass
class CandidateQueue:
    """Queue of candidate queries for Legal review."""
    
    entries: list[CandidateEntry] = field(default_factory=list)
    
    # Thresholds
    FREQUENCY_THRESHOLD = 5          # Add to review if seen >= 5 times
    HIGH_PRIORITY_THRESHOLD = 10     # High priority if seen >= 10 times
    
    def add_candidate(
        self,
        question: str,
        contract_type: str,
        contract_text: str,
        assistant_response: str,
        similarity_to_nearest: float = 0.0,
        nearest_golden_id: Optional[str] = None,
    ) -> CandidateEntry:
        """Add a new candidate or increment existing one."""
        
        # Check if similar question already exists
        existing = self._find_similar(question, contract_type)
        
        if existing:
            # Increment count and update
            existing.occurrence_count += 1
            existing.last_seen = datetime.now().isoformat()
            
            # Update priority based on frequency
            if existing.occurrence_count >= self.HIGH_PRIORITY_THRESHOLD:
                existing.priority = CandidatePriority.HIGH
            elif existing.occurrence_count >= self.FREQUENCY_THRESHOLD:
                existing.priority = CandidatePriority.MEDIUM
            
            return existing
        
        # Create new entry
        entry = CandidateEntry(
            id=f"candidate-{len(self.entries) + 1:04d}",
            question=question,
            contract_type=contract_type,
            contract_text=contract_text,
            assistant_response=assistant_response,
            first_seen=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            similarity_to_nearest=similarity_to_nearest,
            nearest_golden_id=nearest_golden_id,
        )
        
        self.entries.append(entry)
        return entry
    
    def _find_similar(self, question: str, contract_type: str) -> Optional[CandidateEntry]:
        """Find existing candidate with similar question."""
        question_lower = question.lower().strip()
        
        for entry in self.entries:
            if entry.contract_type.lower() == contract_type.lower():
                # Simple exact match for now (production would use similarity)
                if entry.question.lower().strip() == question_lower:
                    return entry
        
        return None
    
    def get_pending_for_review(self) -> list[CandidateEntry]:
        """Get candidates ready for Legal review (met frequency threshold)."""
        return [
            e for e in self.entries
            if e.status == CandidateStatus.PENDING
            and e.occurrence_count >= self.FREQUENCY_THRESHOLD
        ]
    
    def get_high_priority(self) -> list[CandidateEntry]:
        """Get high-priority candidates."""
        return [
            e for e in self.entries
            if e.priority == CandidatePriority.HIGH
            and e.status == CandidateStatus.PENDING
        ]
    
    def approve_candidate(
        self,
        candidate_id: str,
        reviewer: str,
        notes: str = "",
    ) -> Optional[CandidateEntry]:
        """Approve a candidate for addition to golden dataset."""
        for entry in self.entries:
            if entry.id == candidate_id:
                entry.status = CandidateStatus.APPROVED
                entry.reviewer = reviewer
                entry.review_notes = notes
                entry.reviewed_at = datetime.now().isoformat()
                return entry
        return None
    
    def reject_candidate(
        self,
        candidate_id: str,
        reviewer: str,
        notes: str = "",
    ) -> Optional[CandidateEntry]:
        """Reject a candidate."""
        for entry in self.entries:
            if entry.id == candidate_id:
                entry.status = CandidateStatus.REJECTED
                entry.reviewer = reviewer
                entry.review_notes = notes
                entry.reviewed_at = datetime.now().isoformat()
                return entry
        return None
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        total = len(self.entries)
        pending = sum(1 for e in self.entries if e.status == CandidateStatus.PENDING)
        approved = sum(1 for e in self.entries if e.status == CandidateStatus.APPROVED)
        rejected = sum(1 for e in self.entries if e.status == CandidateStatus.REJECTED)
        
        ready_for_review = len(self.get_pending_for_review())
        high_priority = len(self.get_high_priority())
        
        # By contract type
        by_contract = {}
        for entry in self.entries:
            ct = entry.contract_type
            if ct not in by_contract:
                by_contract[ct] = 0
            by_contract[ct] += 1
        
        return {
            "total_candidates": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "ready_for_review": ready_for_review,
            "high_priority": high_priority,
            "by_contract_type": by_contract,
        }
    
    def save(self, filepath: Path) -> None:
        """Save queue to JSON file."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "CandidateQueue":
        """Load queue from JSON file."""
        if not filepath.exists():
            return cls()
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entries = [CandidateEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries=entries)

