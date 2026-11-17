"""Cross-Encoder Reranker module for precision re-scoring."""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-scores candidate query-document pairs using a pretrained CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def _get_model(self):
        if not self.enabled:
            return None
        if self._model is None:
            try:
                import os

                from sentence_transformers import CrossEncoder

                # Check if offline mode or test mode is active
                if (
                    os.environ.get("HF_HUB_OFFLINE") == "1"
                    or os.environ.get("DECISIONIQ_ENV") == "test"
                ):
                    try:
                        self._model = CrossEncoder(
                            self.model_name, device="cpu", local_files_only=True
                        )
                        return self._model
                    except Exception:
                        self.enabled = False
                        return None

                # Attempt local load first, then network
                try:
                    self._model = CrossEncoder(self.model_name, device="cpu", local_files_only=True)
                except Exception:
                    logger.info("Downloading CrossEncoder model: %s", self.model_name)
                    self._model = CrossEncoder(self.model_name, device="cpu")
            except Exception as e:
                logger.warning(
                    "Failed to load CrossEncoder (%s). Falling back to heuristic reranking.",
                    e,
                )
                self.enabled = False
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """Re-rank candidate documents against the query and return top_k with scores."""
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            # Fallback heuristic: sort by original score + lexical term overlap boost
            return self._heuristic_rerank(query, candidates, top_k)

        pairs = [[query, doc.page_content] for doc, _ in candidates]
        try:
            scores = model.predict(pairs)
            scored = [
                (doc, float(score)) for (doc, _), score in zip(candidates, scores, strict=False)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning("Error during CrossEncoder inference: %s. Using fallback.", e)
            return self._heuristic_rerank(query, candidates, top_k)

    def _heuristic_rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        query_words = set(query.lower().split())
        scored = []
        for doc, orig_score in candidates:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words)) / max(1, len(query_words))
            combined_score = 0.5 * orig_score + 0.5 * overlap
            scored.append((doc, combined_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
