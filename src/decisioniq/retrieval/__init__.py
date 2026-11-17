"""Unified retrieval engine supporting Dense, BM25, Hybrid RRF, and CrossEncoder Reranking."""

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.documents import Document

from decisioniq.config import Settings
from decisioniq.config import settings as default_settings
from decisioniq.ingestion.loaders import load_directory
from decisioniq.retrieval.bm25 import BM25Retriever
from decisioniq.retrieval.dense import DenseRetriever
from decisioniq.retrieval.filters import active_version_filter, apply_post_retrieval_filters
from decisioniq.retrieval.hybrid import HybridRetriever
from decisioniq.retrieval.reranker import CrossEncoderReranker
from decisioniq.vectorstores import get_vector_store
from decisioniq.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Configurable enterprise retrieval pipeline with strategy switching and reranking."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        documents: Optional[List[Document]] = None,
        strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        rerank_top_k: int = 3,
        active_only: bool = True,
    ):
        self.vector_store = vector_store
        self.strategy = strategy
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.active_only = active_only

        self.dense_retriever = DenseRetriever(self.vector_store)

        # Lazy load / index BM25 documents if needed
        docs_for_bm25 = documents or []
        if not docs_for_bm25:
            # Fallback to load from data/docs
            try:
                from pathlib import Path

                from decisioniq.ingestion.chunker import DocumentChunker

                raw_docs = load_directory(Path("data/docs"))
                chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
                docs_for_bm25 = chunker.chunk_documents(raw_docs)
            except Exception as e:
                logger.warning("Could not auto-load documents for BM25: %s", e)
                docs_for_bm25 = []

        self.bm25_retriever = BM25Retriever(docs_for_bm25)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever, self.bm25_retriever)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

    def retrieve(
        self,
        query: str,
        strategy_override: Optional[Literal["dense", "bm25", "hybrid", "hybrid_rerank"]] = None,
        top_k_override: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Execute retrieval according to the chosen strategy."""
        strat = strategy_override or self.strategy
        k = top_k_override or self.top_k

        # 1. Fetch initial candidates
        if strat == "dense":
            candidates = self.dense_retriever.retrieve(query, k=k, filter=filter)
        elif strat == "bm25":
            candidates = self.bm25_retriever.retrieve(query, k=k, filter=filter)
        elif strat in ["hybrid", "hybrid_rerank"]:
            candidates = self.hybrid_retriever.retrieve(query, k=k * 2, filter=filter)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strat}")

        # 2. Apply active version post-filter if requested
        if self.active_only and not filter:
            candidates = apply_post_retrieval_filters(candidates, active_version_filter)

        # 3. Apply Reranking if strategy specifies
        if strat == "hybrid_rerank":
            final_k = top_k_override or self.rerank_top_k
            return self.reranker.rerank(query, candidates, top_k=final_k)

        return candidates[:k]


def get_retrieval_pipeline(
    config: Optional[Settings] = None,
    store_type: Optional[Literal["faiss", "chroma"]] = None,
    strategy: Optional[Literal["dense", "bm25", "hybrid", "hybrid_rerank"]] = None,
) -> RetrievalPipeline:
    """Factory creating an end-to-end configured RetrievalPipeline."""
    cfg = config or default_settings
    vstore = get_vector_store(store_type=store_type, config=cfg)
    return RetrievalPipeline(
        vector_store=vstore,
        strategy=strategy or cfg.retrieval_strategy,
        reranker_model=cfg.reranker_model,
        top_k=cfg.top_k,
        rerank_top_k=cfg.rerank_top_k,
    )


__all__ = [
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "RetrievalPipeline",
    "get_retrieval_pipeline",
]
