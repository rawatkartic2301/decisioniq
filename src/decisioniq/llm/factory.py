"""Factory functions to instantiate LLMs and Embedding models based on settings."""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from decisioniq.config import Settings
from decisioniq.config import settings as default_settings
from decisioniq.llm.mock import DeterministicEmbeddings, MockChatModel

logger = logging.getLogger(__name__)


def get_chat_model(
    config: Optional[Settings] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
) -> BaseChatModel:
    """Return a configured chat model instance with automatic fallback to MockChatModel."""
    cfg = config or default_settings
    temp = temperature if temperature is not None else cfg.llm_temperature
    model = model_name or cfg.llm_model

    if cfg.llm_provider == "mock" or cfg.env == "test":
        logger.info("Using MockChatModel (provider=%s, env=%s)", cfg.llm_provider, cfg.env)
        return MockChatModel()

    if cfg.llm_provider == "openai":
        if not cfg.openai_api_key:
            logger.warning("OPENAI_API_KEY not configured. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=temp,
                api_key=cfg.openai_api_key,
            )
        except ImportError:
            logger.warning("langchain-openai not installed. Falling back to MockChatModel.")
            return MockChatModel()

    if cfg.llm_provider == "anthropic":
        if not cfg.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not configured. Falling back to MockChatModel.")
            return MockChatModel()
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model_name=model,
                temperature=temp,
                api_key=cfg.anthropic_api_key,
            )
        except ImportError:
            logger.warning("langchain-anthropic not installed. Falling back to MockChatModel.")
            return MockChatModel()

    if cfg.llm_provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                temperature=temp,
            )
        except Exception as e:
            logger.warning(
                "Failed to connect to Ollama at %s (%s). Falling back to MockChatModel.",
                cfg.ollama_base_url,
                e,
            )
            return MockChatModel()

    logger.warning("Unknown provider %s. Falling back to MockChatModel.", cfg.llm_provider)
    return MockChatModel()


def get_embeddings(
    config: Optional[Settings] = None,
    force_deterministic: bool = False,
) -> Embeddings:
    """Return configured Embeddings instance with fast CPU and offline fallback."""
    cfg = config or default_settings

    if force_deterministic or cfg.env == "test" or cfg.llm_provider == "mock":
        return DeterministicEmbeddings()

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        logger.warning(
            "Could not load HuggingFaceEmbeddings (%s). Falling back to DeterministicEmbeddings.",
            e,
        )
        return DeterministicEmbeddings()
