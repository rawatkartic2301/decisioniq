"""BM25 Okapi sparse retriever module."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def default_tokenizer(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """In-memory BM25 Okapi sparse retriever with metadata filtering."""

    def __init__(
        self,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = list(documents)
        self.corpus = [default_tokenizer(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.corpus, k1=k1, b=b) if self.corpus else None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents using BM25 Okapi with optional metadata filter."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = default_tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Pair documents with scores and filter
        scored_docs: List[Tuple[Document, float]] = []
        for doc, score in zip(self.documents, scores, strict=False):
            if score <= 0.0:
                continue

            if filter:
                match = True
                for fk, fv in filter.items():
                    if doc.metadata.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            scored_docs.append((doc, float(score)))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_docs[:k]

        # Normalize BM25 scores to [0, 1] relative to maximum score in candidate set
        if top_k and top_k[0][1] > 0:
            max_s = top_k[0][1]
            return [(d, s / max_s) for d, s in top_k]
        return top_k
