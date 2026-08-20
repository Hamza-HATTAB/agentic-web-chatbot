import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration management with LangSmith tracing integration.
    """
    groq_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # LangSmith Tracing & Observability (Course Ch. 12)
    langchain_tracing_v2: bool = True
    langchain_project: str = "agentic-web-chatbot"
    langchain_api_key: Optional[str] = None

    default_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_search_results: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def configure_langsmith(self):
        """Set up environment variables for LangSmith telemetry."""
        if self.langchain_api_key or os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            if self.langchain_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key

    def validate_keys(self, model_provider: str = "groq") -> bool:
        """Verify required API keys are populated."""
        if model_provider == "groq" and not self.groq_api_key and not os.getenv("GROQ_API_KEY"):
            return False
        if model_provider == "openai" and not self.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            return False
        return True


settings = Settings()
