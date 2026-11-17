"""Dense vector retrieval module."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from decisioniq.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Performs dense similarity search over a BaseVectorStore."""

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents with normalized similarity scores."""
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
