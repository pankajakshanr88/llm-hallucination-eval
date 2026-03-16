"""Faithfulness evaluator using LLM-as-Judge."""

import json
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult

from ..config import settings


FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI assistant's response is faithful to the provided contract document.

A response is FAITHFUL if every claim or statement in the response can be directly verified from the contract text. 
A response is PARTIALLY FAITHFUL if some claims are supported but others are not.
A response is UNFAITHFUL if it contains claims that contradict or are not supported by the contract.

## Contract Document
{contract_text}

## User Question
{question}

## Assistant Response
{response}

## Your Task
Evaluate the faithfulness of the assistant's response. Consider:
1. Does every factual claim have support in the contract?
2. Are there any invented details (dates, numbers, parties, clauses)?
3. Does the response add external legal information not in the contract?
4. Does the response contradict anything in the contract?

Provide your evaluation as JSON with the following structure:
{{
    "label": "faithful" | "partially_faithful" | "unfaithful",
    "score": <float between 0 and 1>,
    "reasoning": "<explanation of your evaluation>",
    "unsupported_claims": ["<list of any claims not supported by the contract>"]
}}

Only respond with valid JSON, no other text."""


@dataclass
class FaithfulnessResult:
    """Result from faithfulness evaluation."""
    label: str
    score: float
    reasoning: str
    unsupported_claims: list[str]
    
    def to_evaluation_result(self) -> EvaluationResult:
        """Convert to LangSmith EvaluationResult."""
        return EvaluationResult(
            key="faithfulness",
            score=self.score,
            comment=self.reasoning,
            extra={
                "label": self.label,
                "unsupported_claims": self.unsupported_claims,
            }
        )


class FaithfulnessEvaluator:
    """Evaluator that checks if responses are faithful to the contract."""
    
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
    ) -> FaithfulnessResult:
        """Evaluate the faithfulness of a response."""
        prompt = FAITHFULNESS_PROMPT.format(
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
            
            return FaithfulnessResult(
                label=data.get("label", "unfaithful"),
                score=float(data.get("score", 0.0)),
                reasoning=data.get("reasoning", ""),
                unsupported_claims=data.get("unsupported_claims", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for parsing errors
            return FaithfulnessResult(
                label="unfaithful",
                score=0.0,
                reasoning=f"Error parsing evaluation: {str(e)}. Raw: {content[:200]}",
                unsupported_claims=[],
            )
    
    def __call__(self, run, example) -> EvaluationResult:
        """LangSmith evaluator interface."""
        # Extract inputs from the example
        inputs = example.inputs if hasattr(example, 'inputs') else example
        outputs = run.outputs if hasattr(run, 'outputs') else run
        
        question = inputs.get("question", "")
        contract_text = inputs.get("contract_text", "")
        response = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
        
        result = self.evaluate(question, response, contract_text)
        return result.to_evaluation_result()


# Singleton instance for convenience
faithfulness_evaluator = FaithfulnessEvaluator()

