"""Vector store interface and implementations."""

from pathlib import Path
from typing import Literal, Optional

from langchain_core.embeddings import Embeddings

from decisioniq.config import Settings
from decisioniq.config import settings as default_settings
from decisioniq.llm.factory import get_embeddings
from decisioniq.vectorstores.base import BaseVectorStore
from decisioniq.vectorstores.chroma_store import ChromaStore
from decisioniq.vectorstores.faiss_store import FAISSStore


def get_vector_store(
    store_type: Optional[Literal["faiss", "chroma"]] = None,
    embeddings: Optional[Embeddings] = None,
    config: Optional[Settings] = None,
    persist_path: Optional[Path] = None,
) -> BaseVectorStore:
    """Factory to instantiate configured vector store (FAISS or Chroma)."""
    cfg = config or default_settings
    chosen_type = store_type or cfg.vector_store
    embeds = embeddings or get_embeddings(cfg)

    if chosen_type == "faiss":
        target_path = persist_path or cfg.faiss_index_path
        return FAISSStore(embeddings=embeds, persist_path=target_path)
    elif chosen_type == "chroma":
        target_path = persist_path or cfg.chroma_persist_dir
        return ChromaStore(embeddings=embeds, persist_path=target_path)
    else:
        raise ValueError(
            f"Unsupported vector store type: {chosen_type}. Must be 'faiss' or 'chroma'."
        )


__all__ = ["BaseVectorStore", "FAISSStore", "ChromaStore", "get_vector_store"]
