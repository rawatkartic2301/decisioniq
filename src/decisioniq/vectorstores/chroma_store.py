"""Chroma vector store adapter implementing BaseVectorStore."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from decisioniq.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """Encapsulates Chroma collection with directory persistence and metadata filtering."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_path: Optional[Path] = None,
        collection_name: str = "decisioniq_docs",
    ):
        self.embeddings = embeddings
        self.persist_path = persist_path or Path("data/vectorstores/chroma")
        self.collection_name = collection_name
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []

        # Ensure all metadata values are primitive types for Chroma compatibility
        cleaned_docs = []
        ids = []
        for i, doc in enumerate(documents):
            cleaned_metadata = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_metadata[k] = v
                else:
                    cleaned_metadata[k] = str(v)
            cleaned_docs.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))
            ids.append(cleaned_metadata.get("chunk_id", f"doc_chunk_{i}"))

        self.store.add_documents(cleaned_docs, ids=ids)
        self.persist()
        return ids

    def _format_chroma_filter(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not filter_dict:
            return None
        if len(filter_dict) == 1:
            k, v = next(iter(filter_dict.items()))
            return {k: {"$eq": v}}
        # Multi-attribute filter
        conditions = [{k: {"$eq": v}} for k, v in filter_dict.items()]
        return {"$and": conditions}

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        chroma_filter = self._format_chroma_filter(filter)
        return self.store.similarity_search(query, k=k, filter=chroma_filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        chroma_filter = self._format_chroma_filter(filter)
        results = self.store.similarity_search_with_score(query, k=k, filter=chroma_filter)
        # Chroma cosine distance is distance (lower is closer). Normalize to [0, 1] similarity
        converted = []
        for doc, dist in results:
            sim = 1.0 / (1.0 + max(0.0, float(dist)))
            converted.append((doc, sim))
        return converted

    def persist(self) -> None:
        # Modern Chroma auto-persists to directory
        pass

    def count(self) -> int:
        try:
            return self.store._collection.count()
        except Exception:
            return 0
