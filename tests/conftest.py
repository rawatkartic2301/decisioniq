"""Global Pytest fixtures and test configuration."""

import pytest

from decisioniq.config import Settings
from decisioniq.llm.mock import DeterministicEmbeddings, MockChatModel


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure all tests run under test environment settings with mock LLM."""
    monkeypatch.setenv("DECISIONIQ_ENV", "test")
    monkeypatch.setenv("DECISIONIQ_LLM_PROVIDER", "mock")
    monkeypatch.setenv("DECISIONIQ_DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture
def test_settings() -> Settings:
    """Return explicit test configuration."""
    return Settings(
        DECISIONIQ_ENV="test",
        DECISIONIQ_LLM_PROVIDER="mock",
        DECISIONIQ_DATABASE_URL="sqlite:///:memory:",
    )


@pytest.fixture
def mock_llm() -> MockChatModel:
    """Return an instance of MockChatModel."""
    return MockChatModel()


@pytest.fixture
def mock_embeddings() -> DeterministicEmbeddings:
    """Return deterministic embedding generator."""
    return DeterministicEmbeddings(dimension=384)
