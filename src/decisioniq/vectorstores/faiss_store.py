"""FAISS vector store adapter implementing BaseVectorStore."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from decisioniq.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSStore(BaseVectorStore):
    """Encapsulates FAISS index with local disk persistence and metadata filtering."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_path: Optional[Path] = None,
    ):
        self.embeddings = embeddings
        self.persist_path = persist_path or Path("data/vectorstores/faiss")
        self.store: Optional[FAISS] = None
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if self.persist_path.exists() and (self.persist_path / "index.faiss").exists():
            try:
                self.store = FAISS.load_local(
                    folder_path=str(self.persist_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s", self.persist_path)
                return True
            except Exception as e:
                logger.warning("Could not load FAISS index from %s: %s", self.persist_path, e)
        return False

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []

        if self.store is None:
            self.store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.store.add_documents(documents)

        self.persist()
        # Return chunk IDs from metadata
        return [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(documents)]

    def _build_filter_func(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Optional[Callable[[Dict[str, Any]], bool]]:
        if not filter_dict:
            return None

        def matches(metadata: Dict[str, Any]) -> bool:
            for key, val in filter_dict.items():
                if metadata.get(key) != val:
                    return False
            return True

        return matches

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if self.store is None:
            return []
        filter_func = self._build_filter_func(filter)
        return self.store.similarity_search(query, k=k, filter=filter_func)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        if self.store is None:
            return []
        filter_func = self._build_filter_func(filter)
        results = self.store.similarity_search_with_score(query, k=k, filter=filter_func)
        # In FAISS default L2 distance, lower is closer. Convert to pseudo-similarity score in [0, 1]
        converted = []
        for doc, dist in results:
            sim = 1.0 / (1.0 + float(dist))
            converted.append((doc, sim))
        return converted

    def persist(self) -> None:
        if self.store is not None:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.persist_path))
            logger.debug("Persisted FAISS index to %s", self.persist_path)

    def count(self) -> int:
        if self.store is None or self.store.docstore is None:
            return 0
        return len(self.store.docstore._dict)
