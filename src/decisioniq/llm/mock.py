"""Mock LLM and Deterministic Embeddings for fast, offline testing and CPU execution."""

import hashlib
from typing import Any, AsyncIterator, Iterator, List, Optional

import numpy as np
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class DeterministicEmbeddings(Embeddings):
    """Produces reproducible, normalized 384-dimensional dense vectors based on text hash."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        # Hash text to seed random state deterministically
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class MockChatModel(BaseChatModel):
    """Deterministic, offline BaseChatModel for local development and unit tests."""

    canned_response: Optional[str] = None
    stream_delay: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_message = messages[-1].content if messages else ""
        response_text = self._compute_response(str(last_message))
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager=None, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last_message = messages[-1].content if messages else ""
        response_text = self._compute_response(str(last_message))
        words = response_text.split(" ")
        for i, word in enumerate(words):
            chunk_text = word if i == len(words) - 1 else word + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if run_manager:
                run_manager.on_llm_new_token(chunk_text)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop, run_manager=None, **kwargs):
            yield chunk

    def _compute_response(self, prompt_text: str) -> str:
        if self.canned_response:
            return self.canned_response

        # Context-aware deterministic rule fallbacks for testing agent nodes
        lowered = prompt_text.lower()
        if "classify" in lowered or "intent" in lowered:
            if any(
                w in lowered
                for w in [
                    "pet",
                    "cat",
                    "dog",
                    "gym",
                    "scooter",
                    "tesla",
                    "everest",
                    "antarctica",
                    "bitcoin",
                    "scuba",
                ]
            ):
                return '{"intent": "unanswerable", "confidence": 0.95}'
            if any(
                w in lowered
                for w in ["override", "admin_key", "compromised", "drop table", "passwords"]
            ):
                return '{"intent": "adversarial", "confidence": 0.99}'
            if any(w in lowered for w in ["anomal", "fraud", "irregular", "spike"]):
                return '{"intent": "anomaly_detection", "confidence": 0.95}'
            if any(w in lowered for w in ["overtime rule", "what is the overtime"]) and any(
                w in lowered for w in ["how many", "hours did", "march"]
            ):
                return '{"intent": "hybrid_doc_sql", "confidence": 0.90}'
            if any(
                w in lowered
                for w in [
                    "how many",
                    "count",
                    "average",
                    "total",
                    "sum",
                    "highest",
                    "employees are based",
                ]
            ):
                return '{"intent": "sql_analytics", "confidence": 0.92}'
            if any(
                w in lowered
                for w in ["overtime", "leave", "policy", "per diem", "allowance", "probation"]
            ):
                return '{"intent": "policy_rag", "confidence": 0.95}'
            return '{"intent": "policy_rag", "confidence": 0.70}'

        if "grounding" in lowered or "hallucination" in lowered:
            if any(
                w in lowered
                for w in [
                    "pet",
                    "cat",
                    "dog",
                    "gym",
                    "scooter",
                    "tesla",
                    "everest",
                    "antarctica",
                    "bitcoin",
                    "scuba",
                ]
            ):
                return '{"verdict": "ABSTAIN", "reasoning": "The retrieved documents do not contain policy information on this topic."}'
            return '{"verdict": "GROUNDED", "reasoning": "All statements match provided evidence."}'

        if "rewrite" in lowered:
            return "What is the official policy regarding overtime for grade L3 employees?"

        if any(
            w in lowered
            for w in [
                "pet",
                "cat",
                "dog",
                "gym",
                "scooter",
                "tesla",
                "everest",
                "antarctica",
                "bitcoin",
                "scuba",
            ]
        ):
            return "I don't have enough verified information in the official policy documents to answer this question."

        return (
            "Based on the official enterprise policy [Doc: overtime_policy_v2#chunk_1], "
            "grade L3 employees are entitled to 1.5x overtime multiplier for weekend hours."
        )
