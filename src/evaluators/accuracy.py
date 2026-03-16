"""Accuracy scorer using LLM-as-Judge."""

import json
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult

from ..config import settings


ACCURACY_PROMPT = """You are an expert evaluator assessing the accuracy of an AI assistant's response about a legal contract.

Accuracy measures whether the response correctly answers the user's question based on the contract content.

## Contract Document
{contract_text}

## User Question
{question}

## Assistant Response
{response}

## Your Task
Evaluate the accuracy of the response on a scale of 0 to 1:

Consider these factors:
1. **Correctness**: Are the facts stated correctly according to the contract?
2. **Completeness**: Does it answer the full question or just part of it?
3. **Relevance**: Does it address what was actually asked?
4. **Precision**: Are specific details (numbers, dates, parties) accurate?

Scoring guide:
- 1.0: Completely accurate, fully answers the question with correct details
- 0.8-0.9: Mostly accurate with minor omissions
- 0.6-0.7: Partially accurate, some errors or significant omissions
- 0.4-0.5: Mixed accuracy, several errors
- 0.2-0.3: Mostly inaccurate
- 0.0-0.1: Completely inaccurate or does not answer the question

Provide your evaluation as JSON:
{{
    "score": <float between 0 and 1>,
    "correctness": <float 0-1>,
    "completeness": <float 0-1>,
    "relevance": <float 0-1>,
    "precision": <float 0-1>,
    "reasoning": "<explanation of your scoring>",
    "errors": ["<list any errors found in the response>"]
}}

Only respond with valid JSON, no other text."""


@dataclass
class AccuracyResult:
    """Result from accuracy evaluation."""
    score: float
    correctness: float
    completeness: float
    relevance: float
    precision: float
    reasoning: str
    errors: list[str]
    
    def to_evaluation_result(self) -> EvaluationResult:
        """Convert to LangSmith EvaluationResult."""
        return EvaluationResult(
            key="accuracy",
            score=self.score,
            comment=self.reasoning,
            extra={
                "correctness": self.correctness,
                "completeness": self.completeness,
                "relevance": self.relevance,
                "precision": self.precision,
                "errors": self.errors,
            }
        )


class AccuracyScorer:
    """Evaluator that scores the accuracy of responses."""
    
    def __init__(self, model: Optional[str] = None):
        self.model_name = model or settings.eval_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
        )
    
    def score(
        self,
        question: str,
        response: str,
        contract_text: str,
    ) -> AccuracyResult:
        """Score the accuracy of a response."""
        prompt = ACCURACY_PROMPT.format(
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
            
            return AccuracyResult(
                score=float(data.get("score", 0.0)),
                correctness=float(data.get("correctness", 0.0)),
                completeness=float(data.get("completeness", 0.0)),
                relevance=float(data.get("relevance", 0.0)),
                precision=float(data.get("precision", 0.0)),
                reasoning=data.get("reasoning", ""),
                errors=data.get("errors", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for parsing errors
            return AccuracyResult(
                score=0.0,
                correctness=0.0,
                completeness=0.0,
                relevance=0.0,
                precision=0.0,
                reasoning=f"Error parsing evaluation: {str(e)}. Raw: {content[:200]}",
                errors=[],
            )
    
    def __call__(self, run, example) -> EvaluationResult:
        """LangSmith evaluator interface."""
        # Extract inputs from the example
        inputs = example.inputs if hasattr(example, 'inputs') else example
        outputs = run.outputs if hasattr(run, 'outputs') else run
        
        question = inputs.get("question", "")
        contract_text = inputs.get("contract_text", "")
        response = outputs.get("response", "") if isinstance(outputs, dict) else str(outputs)
        
        result = self.score(question, response, contract_text)
        return result.to_evaluation_result()


# Singleton instance for convenience
accuracy_scorer = AccuracyScorer()

