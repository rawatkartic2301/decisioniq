"""Hybrid Retrieval combining Dense and Sparse BM25 via Reciprocal Rank Fusion (RRF)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from decisioniq.retrieval.bm25 import BM25Retriever
from decisioniq.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """Merge multiple ranked lists of (Document, score) using Reciprocal Rank Fusion."""
    rrf_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _orig_score) in enumerate(ranked_list):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:100])
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc

            # RRF formula: 1 / (k + rank) where rank is 1-indexed
            contribution = 1.0 / (rrf_k + (rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

    # Sort descending by fused RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: List[Tuple[Document, float]] = []
    for doc_id, score in sorted_items[:top_k]:
        results.append((doc_lookup[doc_id], score))

    return results


class HybridRetriever:
    """Combines Dense and BM25 retrievers with configurable Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        candidate_pool_size: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Fetch candidates from dense and BM25 retrievers and combine using RRF."""
        pool_k = candidate_pool_size or (k * 3)

        dense_candidates = self.dense_retriever.retrieve(query, k=pool_k, filter=filter)
        bm25_candidates = self.bm25_retriever.retrieve(query, k=pool_k, filter=filter)

        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_candidates, bm25_candidates],
            rrf_k=self.rrf_k,
            top_k=k,
        )
        return fused
