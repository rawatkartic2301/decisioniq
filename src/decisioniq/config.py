"""Configuration management for DecisionIQ using Pydantic Settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General & Logging
    env: Literal["development", "test", "production"] = Field(
        default="development", validation_alias="DECISIONIQ_ENV"
    )
    log_level: str = Field(default="INFO", validation_alias="DECISIONIQ_LOG_LEVEL")

    # Vector Store & Retrieval
    vector_store: Literal["faiss", "chroma"] = Field(
        default="faiss", validation_alias="DECISIONIQ_VECTOR_STORE"
    )
    retrieval_strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid_rerank", validation_alias="DECISIONIQ_RETRIEVAL_STRATEGY"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="DECISIONIQ_EMBEDDING_MODEL",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="DECISIONIQ_RERANKER_MODEL",
    )
    top_k: int = Field(default=5, validation_alias="DECISIONIQ_TOP_K")
    rerank_top_k: int = Field(default=3, validation_alias="DECISIONIQ_RERANK_TOP_K")

    # LLM Provider Configuration
    llm_provider: Literal["mock", "openai", "anthropic", "ollama"] = Field(
        default="mock", validation_alias="DECISIONIQ_LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="DECISIONIQ_LLM_MODEL")
    llm_temperature: float = Field(default=0.0, validation_alias="DECISIONIQ_LLM_TEMPERATURE")

    # API Keys & Endpoints (optional)
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")

    # Database & Storage
    database_url: str = Field(
        default="sqlite:///data/db/decisioniq.db",
        validation_alias="DECISIONIQ_DATABASE_URL",
    )
    faiss_index_path: Path = Field(
        default=Path("data/vectorstores/faiss"),
        validation_alias="DECISIONIQ_FAISS_INDEX_PATH",
    )
    chroma_persist_dir: Path = Field(
        default=Path("data/vectorstores/chroma"),
        validation_alias="DECISIONIQ_CHROMA_PERSIST_DIR",
    )

    # Guardrails & Safety
    api_key: str = Field(
        default="decisioniq_dev_api_key_12345", validation_alias="DECISIONIQ_API_KEY"
    )
    rate_limit_per_minute: int = Field(
        default=60, validation_alias="DECISIONIQ_RATE_LIMIT_PER_MINUTE"
    )
    sql_query_timeout_seconds: int = Field(
        default=5, validation_alias="DECISIONIQ_SQL_QUERY_TIMEOUT_SECONDS"
    )
    sql_max_limit: int = Field(default=100, validation_alias="DECISIONIQ_SQL_MAX_LIMIT")

    # Observability & MLflow
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="decisioniq", validation_alias="LANGCHAIN_PROJECT")
    mlflow_tracking_uri: str = Field(default="mlruns", validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="decisioniq-eval", validation_alias="MLFLOW_EXPERIMENT_NAME"
    )


# Global settings singleton
settings = Settings()
