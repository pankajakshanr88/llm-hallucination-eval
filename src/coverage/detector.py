"""Coverage detection - determines if a query has golden dataset coverage."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re

from ..dataset import GoldenDataset, DatasetEntry, load_golden_dataset


class CoverageStatus(str, Enum):
    """Status of coverage for a query."""
    GOLDEN_MATCH = "golden_match"           # Exact or high-similarity match found
    PARTIAL_MATCH = "partial_match"         # Same contract type, different question
    NO_COVERAGE = "no_coverage"             # No matching contract type
    OUT_OF_SCOPE = "out_of_scope"           # Query not answerable from any contract


@dataclass
class CoverageResult:
    """Result of coverage detection for a query."""
    status: CoverageStatus
    confidence: float                        # 0-1 confidence in the match
    matched_entry: Optional[DatasetEntry]    # The matched golden entry if found
    contract_type_match: bool                # Whether contract type matched
    question_category_match: bool            # Whether question category matched
    similarity_score: float                  # Semantic similarity to nearest golden
    recommendation: str                      # What to do with this query
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "matched_entry_id": self.matched_entry.id if self.matched_entry else None,
            "contract_type_match": self.contract_type_match,
            "question_category_match": self.question_category_match,
            "similarity_score": self.similarity_score,
            "recommendation": self.recommendation,
        }


class CoverageDetector:
    """Detects whether a query has coverage in the golden dataset."""
    
    # Keywords that indicate out-of-scope queries
    OUT_OF_SCOPE_PATTERNS = [
        r"should i sign",
        r"is this (a )?good (deal|contract)",
        r"will this hold up in court",
        r"can i sue",
        r"what should i do",
        r"is this legal",
        r"is this enforceable",
        r"recommend",
        r"advise",
        r"opinion",
    ]
    
    # Similarity thresholds
    HIGH_SIMILARITY_THRESHOLD = 0.8
    PARTIAL_SIMILARITY_THRESHOLD = 0.5
    
    def __init__(self, golden_dataset: Optional[GoldenDataset] = None):
        self.golden_dataset = golden_dataset or load_golden_dataset()
        self._build_index()
    
    def _build_index(self):
        """Build lookup indices for faster matching."""
        # Index by contract type
        self.by_contract_type = {}
        for entry in self.golden_dataset:
            ct = entry.contract_type.lower()
            if ct not in self.by_contract_type:
                self.by_contract_type[ct] = []
            self.by_contract_type[ct].append(entry)
        
        # Index by question keywords
        self.question_keywords = {}
        for entry in self.golden_dataset:
            keywords = self._extract_keywords(entry.question)
            for kw in keywords:
                if kw not in self.question_keywords:
                    self.question_keywords[kw] = []
                self.question_keywords[kw].append(entry)
    
    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text."""
        # Simple keyword extraction
        stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'does', 'do', 'can', 'this', 'my', 'i'}
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}
    
    def _is_out_of_scope(self, question: str) -> bool:
        """Check if question is asking for legal advice (out of scope)."""
        question_lower = question.lower()
        for pattern in self.OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, question_lower):
                return True
        return False
    
    def _calculate_similarity(self, query: str, golden_question: str) -> float:
        """Calculate simple keyword-based similarity (production would use embeddings)."""
        query_keywords = self._extract_keywords(query)
        golden_keywords = self._extract_keywords(golden_question)
        
        if not query_keywords or not golden_keywords:
            return 0.0
        
        intersection = query_keywords & golden_keywords
        union = query_keywords | golden_keywords
        
        return len(intersection) / len(union) if union else 0.0
    
    def detect(
        self,
        question: str,
        contract_type: str,
        contract_text: str = "",
    ) -> CoverageResult:
        """Detect coverage status for a query.
        
        Args:
            question: The user's question
            contract_type: Type of contract being queried
            contract_text: The actual contract text (optional)
            
        Returns:
            CoverageResult with status and recommendations
        """
        # Check if out of scope (asking for legal advice)
        if self._is_out_of_scope(question):
            return CoverageResult(
                status=CoverageStatus.OUT_OF_SCOPE,
                confidence=0.9,
                matched_entry=None,
                contract_type_match=False,
                question_category_match=False,
                similarity_score=0.0,
                recommendation="Route to guardrail response. Do not include in hallucination metrics."
            )
        
        # Check for contract type match
        contract_type_lower = contract_type.lower()
        type_matches = []
        
        for ct, entries in self.by_contract_type.items():
            if ct in contract_type_lower or contract_type_lower in ct:
                type_matches.extend(entries)
        
        contract_type_match = len(type_matches) > 0
        
        # If no contract type match, it's no coverage
        if not contract_type_match:
            return CoverageResult(
                status=CoverageStatus.NO_COVERAGE,
                confidence=0.8,
                matched_entry=None,
                contract_type_match=False,
                question_category_match=False,
                similarity_score=0.0,
                recommendation="New contract type. Add to candidate queue for Legal review."
            )
        
        # Find best matching question within contract type
        best_match = None
        best_similarity = 0.0
        
        for entry in type_matches:
            sim = self._calculate_similarity(question, entry.question)
            if sim > best_similarity:
                best_similarity = sim
                best_match = entry
        
        # Determine status based on similarity
        if best_similarity >= self.HIGH_SIMILARITY_THRESHOLD:
            return CoverageResult(
                status=CoverageStatus.GOLDEN_MATCH,
                confidence=best_similarity,
                matched_entry=best_match,
                contract_type_match=True,
                question_category_match=True,
                similarity_score=best_similarity,
                recommendation="Full evaluation with golden reference."
            )
        elif best_similarity >= self.PARTIAL_SIMILARITY_THRESHOLD:
            return CoverageResult(
                status=CoverageStatus.PARTIAL_MATCH,
                confidence=best_similarity,
                matched_entry=best_match,
                contract_type_match=True,
                question_category_match=False,
                similarity_score=best_similarity,
                recommendation="Rubric-only evaluation. Consider adding to golden set if frequent."
            )
        else:
            return CoverageResult(
                status=CoverageStatus.NO_COVERAGE,
                confidence=0.7,
                matched_entry=None,
                contract_type_match=True,
                question_category_match=False,
                similarity_score=best_similarity,
                recommendation="New question pattern. Log to candidate queue."
            )
    
    def get_coverage_stats(self) -> dict:
        """Get statistics about golden dataset coverage."""
        contract_types = list(self.by_contract_type.keys())
        
        stats = {
            "total_golden_entries": len(self.golden_dataset),
            "contract_types_covered": len(contract_types),
            "contract_types": {},
        }
        
        for ct, entries in self.by_contract_type.items():
            categories = {}
            for entry in entries:
                cat = entry.question_category.value
                categories[cat] = categories.get(cat, 0) + 1
            
            stats["contract_types"][ct] = {
                "total_questions": len(entries),
                "categories": categories,
            }
        
        return stats

