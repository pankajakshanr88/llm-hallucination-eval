"""Hallucination detector using LLM-as-Judge."""

import json
from dataclasses import dataclass, field
from typing import Optional

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult

from ..config import settings


HALLUCINATION_PROMPT = """You are an expert hallucination detector for a legal document assistant.

A HALLUCINATION occurs when the assistant generates content NOT SUPPORTED by the retrieved contract context.

Examples of hallucinations include:
- Inventing a clause that doesn't exist in the contract
- Changing numbers, dates, or parties from what the contract states
- Adding obligations or rights not found in the contract
- Generalizing or making claims outside the scope of the provided text
- Providing jurisdiction-specific legal information not in the contract
- Making recommendations or giving legal advice

## Contract Document (Ground Truth)
{contract_text}

## User Question
{question}

## Assistant Response to Evaluate
{response}

## Your Task
Analyze the assistant's response and identify ANY hallucinations.

For each statement in the response, verify it against the contract. Flag any content that:
1. Cannot be verified from the contract text
2. Contradicts the contract
3. Adds external legal information
4. Makes unfounded inferences

Provide your evaluation as JSON:
{{
    "contains_hallucination": true | false,
    "hallucination_count": <number of distinct hallucinations>,
    "severity": "none" | "minor" | "major" | "critical",
    "hallucinated_claims": [
        {{
            "claim": "<the specific hallucinated statement>",
            "reason": "<why this is a hallucination>",
            "severity": "minor" | "major" | "critical"
        }}
    ],
    "score": <float 0-1, where 1 = no hallucinations, 0 = severe hallucinations>,
    "reasoning": "<overall assessment>"
}}

Severity levels:
- minor: Minor embellishment that doesn't materially affect understanding
- major: Incorrect information that could mislead the user
- critical: Fundamental error like inventing clauses or reversing contract terms

Only respond with valid JSON, no other text."""


@dataclass
class HallucinatedClaim:
    """A single hallucinated claim identified in the response."""
    claim: str
    reason: str
    severity: str


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""
    contains_hallucination: bool
    hallucination_count: int
    severity: str
    hallucinated_claims: list[HallucinatedClaim] = field(default_factory=list)
    score: float = 1.0
    reasoning: str = ""
    
    def to_evaluation_result(self) -> EvaluationResult:
        """Convert to LangSmith EvaluationResult."""
        return EvaluationResult(
            key="hallucination",
            score=self.score,
            comment=self.reasoning,
            extra={
                "contains_hallucination": self.contains_hallucination,
                "hallucination_count": self.hallucination_count,
                "severity": self.severity,
                "hallucinated_claims": [
                    {"claim": c.claim, "reason": c.reason, "severity": c.severity}
                    for c in self.hallucinated_claims
                ],
            }
        )


class HallucinationDetector:
    """Detector that identifies hallucinations in assistant responses."""
    
    def __init__(self, model: Optional[str] = None):
        self.model_name = model or settings.eval_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
        )
    
    def detect(
        self,
        question: str,
        response: str,
        contract_text: str,
    ) -> HallucinationResult:
        """Detect hallucinations in a response."""
        prompt = HALLUCINATION_PROMPT.format(
            contract_text=contract_text,
            question=question,
            response=response,
        )
        
        result = self.llm.invoke(prompt)
        content = result.content
        
        # Parse JSON response
        try:
            # Handle markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            # Parse hallucinated claims
            claims = []
            for claim_data in data.get("hallucinated_claims", []):
                claims.append(HallucinatedClaim(
                    claim=claim_data.get("claim", ""),
                    reason=claim_data.get("reason", ""),
                    severity=claim_data.get("severity", "minor"),
                ))
            
            return HallucinationResult(
                contains_hallucination=data.get("contains_hallucination", False),
                hallucination_count=data.get("hallucination_count", 0),
                severity=data.get("severity", "none"),
                hallucinated_claims=claims,
                score=float(data.get("score", 1.0)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for parsing errors
            return HallucinationResult(
                contains_hallucination=True,
                hallucination_count=0,
                severity="unknown",
                score=0.0,
                reasoning=f"Error parsing evaluation: {str(e)}. Raw: {content[:200]}",
            )
    
    def __call__(self, run, example) -> EvaluationResult:
        """LangSmith evaluator interface."""
        # Extract inputs from the example
        inputs = example.inputs if hasattr(example, 'inputs') else example
        outputs = run.outputs if hasattr(run, 'outputs') else run
        
        question = inputs.get("question", "")
        contract_text = inputs.get("contract_text", "")
        response = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
        
        result = self.detect(question, response, contract_text)
        return result.to_evaluation_result()


# Singleton instance for convenience
hallucination_detector = HallucinationDetector()

