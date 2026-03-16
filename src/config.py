"""LangSmith and application configuration."""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Also load from config.ini (if exists)
config_path = Path(__file__).parent.parent / "config.ini"
if config_path.exists():
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Set from config.ini if not already in environment
    if config.has_section("langsmith"):
        if not os.getenv("LANGCHAIN_API_KEY") and config.get("langsmith", "api_key", fallback=""):
            os.environ["LANGCHAIN_API_KEY"] = config.get("langsmith", "api_key")
        if config.get("langsmith", "project", fallback=""):
            os.environ["LANGCHAIN_PROJECT"] = config.get("langsmith", "project")
    
    if config.has_section("openai"):
        if not os.getenv("OPENAI_API_KEY"):
            api_key = config.get("openai", "api_key", fallback="")
            if api_key and api_key != "YOUR_OPENAI_API_KEY_HERE":
                os.environ["OPENAI_API_KEY"] = api_key


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # LangSmith settings
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_tracing: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    langchain_project: str = os.getenv(
        "LANGCHAIN_PROJECT", 
        "document-assistant-hallucination-tracking"
    )
    
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    eval_model: str = os.getenv("EVAL_MODEL", "gpt-4o")
    
    # Thresholds from spec
    hallucination_rate_threshold: float = 0.02  # ≤ 2% for launch
    hallucination_rate_warning: float = 0.10    # > 10% is critical
    accuracy_threshold: float = 0.95            # ≥ 95% for launch
    judge_agreement_threshold: float = 0.80     # ≥ 80% for launch
    
    def __post_init__(self):
        """Set environment variables for LangSmith."""
        if self.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = self.langchain_tracing
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
    
    def validate(self) -> list[str]:
        """Validate that required settings are present."""
        errors = []
        if not self.langchain_api_key:
            errors.append("LANGCHAIN_API_KEY is not set")
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is not set")
        return errors
    
    def __str__(self) -> str:
        """String representation hiding sensitive values."""
        return (
            f"Settings(\n"
            f"  langchain_api_key={'*' * 8 if self.langchain_api_key else 'NOT SET'},\n"
            f"  langchain_project={self.langchain_project},\n"
            f"  openai_api_key={'*' * 8 if self.openai_api_key else 'NOT SET'},\n"
            f"  eval_model={self.eval_model},\n"
            f"  hallucination_rate_threshold={self.hallucination_rate_threshold},\n"
            f"  accuracy_threshold={self.accuracy_threshold},\n"
            f"  judge_agreement_threshold={self.judge_agreement_threshold}\n"
            f")"
        )


# Global settings instance
settings = Settings()

