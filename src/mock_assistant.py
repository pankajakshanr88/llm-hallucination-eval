"""Mock Document Assistant for testing the evaluation pipeline.

This mock assistant simulates the real Document Assistant behavior.
It can be configured to produce responses with varying levels of hallucination
to test the evaluation system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import random

from langchain_openai import ChatOpenAI

from .config import settings


class ResponseMode(str, Enum):
    """Mode for mock assistant response generation."""
    FAITHFUL = "faithful"           # Always generate faithful responses
    HALLUCINATING = "hallucinating" # Always include some hallucination
    MIXED = "mixed"                 # Randomly mix faithful and hallucinating
    LLM = "llm"                     # Use actual LLM (most realistic)


ASSISTANT_SYSTEM_PROMPT = """You are a Document Assistant that helps users understand legal contracts in plain English.

SCOPE: Answer questions ONLY using information explicitly contained in the provided contract document.

WHAT YOU DO:
- Identify and quote specific clauses
- Summarize contract provisions
- Point out where topics are/aren't addressed
- Reference specific section numbers

WHAT YOU DON'T DO:
- Provide legal advice or recommendations
- Interpret legal implications
- Answer using external legal knowledge
- Suggest what actions to take

RESPONSE FORMAT:
- Quote directly from contract with section references
- If information isn't in the contract, clearly state: "This is not addressed in the provided contract"
- Always cite: [Section X.X] for every reference

Contract Document:
{contract_text}

User Question: {question}

Provide a helpful response based only on the contract above."""


HALLUCINATING_SYSTEM_PROMPT = """You are a Document Assistant that helps users understand legal contracts.

IMPORTANT: For testing purposes, you should include ONE subtle hallucination in your response.
This means adding ONE piece of information that is NOT in the contract, such as:
- A made-up legal requirement or statute
- An invented penalty or consequence
- A number or date that differs from the contract
- An external legal concept not mentioned in the document

The hallucination should be subtle and mixed with otherwise accurate information.

Contract Document:
{contract_text}

User Question: {question}

Provide a response that is mostly accurate but contains one subtle hallucination for testing."""


@dataclass
class AssistantResponse:
    """Response from the mock assistant."""
    response: str
    mode: ResponseMode
    intentional_hallucination: bool = False


class MockDocumentAssistant:
    """Mock Document Assistant for testing evaluation pipeline.
    
    This assistant can operate in different modes:
    - FAITHFUL: Uses LLM with strict grounding instructions
    - HALLUCINATING: Uses LLM instructed to include hallucinations
    - MIXED: Randomly switches between faithful and hallucinating
    - LLM: Uses the standard assistant prompt (most realistic)
    """
    
    def __init__(
        self,
        mode: ResponseMode = ResponseMode.LLM,
        model: Optional[str] = None,
        hallucination_rate: float = 0.3,
    ):
        """Initialize the mock assistant.
        
        Args:
            mode: Response generation mode
            model: OpenAI model to use
            hallucination_rate: For MIXED mode, probability of hallucinating
        """
        self.mode = mode
        self.model_name = model or settings.eval_model
        self.hallucination_rate = hallucination_rate
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.3,
        )
    
    def respond(
        self,
        question: str,
        contract_text: str,
        contract_type: str = "",
    ) -> AssistantResponse:
        """Generate a response to a question about a contract.
        
        Args:
            question: User's question
            contract_text: The contract document
            contract_type: Type of contract (for context)
            
        Returns:
            AssistantResponse with the generated response
        """
        # Determine which mode to use for this response
        effective_mode = self.mode
        intentional_hallucination = False
        
        if self.mode == ResponseMode.MIXED:
            if random.random() < self.hallucination_rate:
                effective_mode = ResponseMode.HALLUCINATING
                intentional_hallucination = True
            else:
                effective_mode = ResponseMode.FAITHFUL
        elif self.mode == ResponseMode.HALLUCINATING:
            intentional_hallucination = True
        
        # Select prompt based on mode
        if effective_mode == ResponseMode.HALLUCINATING:
            prompt = HALLUCINATING_SYSTEM_PROMPT.format(
                contract_text=contract_text,
                question=question,
            )
        else:
            prompt = ASSISTANT_SYSTEM_PROMPT.format(
                contract_text=contract_text,
                question=question,
            )
        
        # Generate response
        result = self.llm.invoke(prompt)
        
        return AssistantResponse(
            response=result.content,
            mode=effective_mode,
            intentional_hallucination=intentional_hallucination,
        )
    
    def __call__(self, inputs: dict) -> dict:
        """LangSmith-compatible interface.
        
        Args:
            inputs: Dict with 'question', 'contract_text', and optionally 'contract_type'
            
        Returns:
            Dict with 'response' key
        """
        question = inputs.get("question", "")
        contract_text = inputs.get("contract_text", "")
        contract_type = inputs.get("contract_type", "")
        
        result = self.respond(question, contract_text, contract_type)
        
        return {
            "response": result.response,
            "mode": result.mode.value,
            "intentional_hallucination": result.intentional_hallucination,
        }


class StaticMockAssistant:
    """A simple mock assistant that returns pre-defined responses.
    
    Useful for testing the evaluation pipeline without LLM calls.
    """
    
    def __init__(self, responses: Optional[dict[str, str]] = None):
        """Initialize with optional pre-defined responses.
        
        Args:
            responses: Dict mapping question to response
        """
        self.responses = responses or {}
        self.default_response = "I can help you understand this contract. Based on the document provided, I can answer questions about its contents."
    
    def add_response(self, question: str, response: str) -> None:
        """Add a pre-defined response for a question."""
        self.responses[question] = response
    
    def respond(
        self,
        question: str,
        contract_text: str,
        contract_type: str = "",
    ) -> AssistantResponse:
        """Return pre-defined response or default."""
        response = self.responses.get(question, self.default_response)
        return AssistantResponse(
            response=response,
            mode=ResponseMode.FAITHFUL,
            intentional_hallucination=False,
        )
    
    def __call__(self, inputs: dict) -> dict:
        """LangSmith-compatible interface."""
        question = inputs.get("question", "")
        contract_text = inputs.get("contract_text", "")
        contract_type = inputs.get("contract_type", "")
        
        result = self.respond(question, contract_text, contract_type)
        
        return {
            "response": result.response,
            "mode": result.mode.value,
            "intentional_hallucination": result.intentional_hallucination,
        }


# Pre-configured instances
faithful_assistant = MockDocumentAssistant(mode=ResponseMode.FAITHFUL)
hallucinating_assistant = MockDocumentAssistant(mode=ResponseMode.HALLUCINATING)
mixed_assistant = MockDocumentAssistant(mode=ResponseMode.MIXED, hallucination_rate=0.3)

