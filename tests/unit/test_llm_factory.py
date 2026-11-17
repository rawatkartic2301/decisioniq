"""Unit tests for LLM factory and mock implementations."""

import pytest
from langchain_core.messages import HumanMessage

from decisioniq.config import Settings
from decisioniq.llm.factory import get_chat_model, get_embeddings
from decisioniq.llm.mock import DeterministicEmbeddings, MockChatModel


def test_deterministic_embeddings_dimension_and_norm():
    embedder = DeterministicEmbeddings(dimension=384)
    vec = embedder.embed_query("What is the overtime policy for L3?")
    assert len(vec) == 384
    # Check unit norm approximately
    norm = sum(x**2 for x in vec) ** 0.5
    assert pytest.approx(norm, 0.001) == 1.0

    # Ensure deterministic output for same string
    vec2 = embedder.embed_query("What is the overtime policy for L3?")
    assert vec == vec2

    # Different text yields different vector
    vec_diff = embedder.embed_query("Different query text")
    assert vec != vec_diff


def test_mock_chat_model_invoke():
    model = MockChatModel()
    response = model.invoke([HumanMessage(content="What is the policy for leave?")])
    assert response.content is not None
    assert len(response.content) > 0


def test_mock_chat_model_streaming():
    model = MockChatModel()
    chunks = list(model.stream([HumanMessage(content="Hello world")]))
    assert len(chunks) > 0
    full_text = "".join(c.content for c in chunks)
    assert len(full_text) > 0


def test_llm_factory_fallback():
    # In test env with no keys, get_chat_model should safely return MockChatModel
    cfg = Settings(DECISIONIQ_ENV="test", DECISIONIQ_LLM_PROVIDER="openai")
    model = get_chat_model(cfg)
    assert isinstance(model, MockChatModel)

    embedder = get_embeddings(cfg, force_deterministic=True)
    assert isinstance(embedder, DeterministicEmbeddings)
