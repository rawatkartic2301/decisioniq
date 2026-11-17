"""Abstract base vector store interface for DecisionIQ."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Abstract interface defining required vector store operations."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add a batch of langchain Document objects to the index and return IDs."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve the top-k most similar documents for a query."""
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents along with relevance/distance scores."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in store."""
        pass
