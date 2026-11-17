"""Unit tests for configuration loading and validation."""

from decisioniq.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.env in ["development", "test", "production"]
    assert settings.vector_store in ["faiss", "chroma"]
    assert settings.retrieval_strategy in ["dense", "bm25", "hybrid", "hybrid_rerank"]
    assert settings.top_k > 0
    assert settings.sql_max_limit == 100


def test_override_settings(monkeypatch):
    monkeypatch.setenv("DECISIONIQ_VECTOR_STORE", "chroma")
    monkeypatch.setenv("DECISIONIQ_TOP_K", "10")
    monkeypatch.setenv("DECISIONIQ_ENV", "production")

    settings = Settings()
    assert settings.vector_store == "chroma"
    assert settings.top_k == 10
    assert settings.env == "production"
