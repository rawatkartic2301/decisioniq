"""Document retrieval tool wrapping the unified RetrievalPipeline."""

from typing import Any, Dict, Optional

from decisioniq.retrieval import RetrievalPipeline, get_retrieval_pipeline


class DocRetrieverTool:
    """Agent tool that performs hybrid retrieval with reranking and active policy filtering."""

    def __init__(self, pipeline: Optional[RetrievalPipeline] = None):
        self.pipeline = pipeline or get_retrieval_pipeline()

    def run(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Execute document retrieval and return structured chunk results."""
        scored_candidates = self.pipeline.retrieve(query, top_k_override=top_k)
        docs = [doc for doc, _ in scored_candidates]
        chunk_ids = [doc.metadata.get("chunk_id", "doc") for doc in docs]

        return {
            "query": query,
            "count": len(docs),
            "chunk_ids": chunk_ids,
            "documents": docs,
            "scored_candidates": scored_candidates,
        }
