"""Groundedness evaluator for queries without golden dataset coverage.

This evaluator is used when there's no golden reference to compare against.
It checks if the response is grounded in the contract text without needing
a pre-labeled expected answer.
"""

import json
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult

from ..config import settings


GROUNDEDNESS_PROMPT = """You are an evaluator checking if an AI assistant's response is grounded in the provided contract document.

Unlike accuracy evaluation, you do NOT have a reference answer to compare against.
Your job is to verify that every claim in the response can be traced back to the contract.

## Contract Document
{contract_text}

## User Question
{question}

## Assistant Response
{response}

## Your Task

Evaluate the response on these criteria:

1. **Groundedness** (0-1): Is every factual claim supported by the contract?
2. **Relevance** (0-1): Does the response address the user's question?
3. **Clarity** (0-1): Is the response clear and well-organized?
4. **Tone** (0-1): Is the tone professional and appropriate?
5. **Guardrail Compliance** (0-1): Does it avoid giving legal advice?

Also identify:
- Any claims that cannot be verified from the contract
- Whether the response appropriately declines to answer if the info isn't in the contract

Provide your evaluation as JSON:
{{
    "groundedness": <float 0-1>,
    "relevance": <float 0-1>,
    "clarity": <float 0-1>,
    "tone": <float 0-1>,
    "guardrail_compliance": <float 0-1>,
    "overall_score": <float 0-1>,
    "unverifiable_claims": ["<list of claims not in contract>"],
    "appropriate_abstention": <true if correctly declined to answer, false otherwise>,
    "reasoning": "<explanation>"
}}

Only respond with valid JSON, no other text."""


@dataclass
class GroundednessResult:
    """Result from groundedness evaluation (no golden reference needed)."""
    groundedness: float
    relevance: float
    clarity: float
    tone: float
    guardrail_compliance: float
    overall_score: float
    unverifiable_claims: list[str]
    appropriate_abstention: bool
    reasoning: str
    
    def to_evaluation_result(self) -> EvaluationResult:
        """Convert to LangSmith EvaluationResult."""
        return EvaluationResult(
            key="groundedness",
            score=self.overall_score,
            comment=self.reasoning,
            extra={
                "groundedness": self.groundedness,
                "relevance": self.relevance,
                "clarity": self.clarity,
                "tone": self.tone,
                "guardrail_compliance": self.guardrail_compliance,
                "unverifiable_claims": self.unverifiable_claims,
                "appropriate_abstention": self.appropriate_abstention,
            }
        )


class GroundednessEvaluator:
    """Evaluator for responses without golden dataset coverage.
    
    This is a rubric-only evaluator that doesn't need a reference answer.
    It checks if the response is grounded in the contract and follows guidelines.
    """
    
    def __init__(self, model: Optional[str] = None):
        self.model_name = model or settings.eval_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
        )
    
    def evaluate(
        self,
        question: str,
        response: str,
        contract_text: str,
    ) -> GroundednessResult:
        """Evaluate groundedness of a response without golden reference."""
        prompt = GROUNDEDNESS_PROMPT.format(
            contract_text=contract_text,
            question=question,
            response=response,
        )
        
        result = self.llm.invoke(prompt)
        content = result.content
        
        # Parse JSON response
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            return GroundednessResult(
                groundedness=float(data.get("groundedness", 0.0)),
                relevance=float(data.get("relevance", 0.0)),
                clarity=float(data.get("clarity", 0.0)),
                tone=float(data.get("tone", 0.0)),
                guardrail_compliance=float(data.get("guardrail_compliance", 0.0)),
                overall_score=float(data.get("overall_score", 0.0)),
                unverifiable_claims=data.get("unverifiable_claims", []),
                appropriate_abstention=data.get("appropriate_abstention", False),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return GroundednessResult(
                groundedness=0.0,
                relevance=0.0,
                clarity=0.0,
                tone=0.0,
                guardrail_compliance=0.0,
                overall_score=0.0,
                unverifiable_claims=[],
                appropriate_abstention=False,
                reasoning=f"Error parsing evaluation: {str(e)}",
            )
    
    def __call__(self, run, example) -> EvaluationResult:
        """LangSmith evaluator interface."""
        inputs = example.inputs if hasattr(example, 'inputs') else example
        outputs = run.outputs if hasattr(run, 'outputs') else run
        
        question = inputs.get("question", "")
        contract_text = inputs.get("contract_text", "")
        response = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
        
        result = self.evaluate(question, response, contract_text)
        return result.to_evaluation_result()


# Singleton instance
groundedness_evaluator = GroundednessEvaluator()

