"""Baseline LangChain LCEL RAG chain for benchmarking and direct comparison."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from decisioniq.config import Settings
from decisioniq.config import settings as default_settings
from decisioniq.llm.factory import get_chat_model
from decisioniq.retrieval import RetrievalPipeline, get_retrieval_pipeline

logger = logging.getLogger(__name__)

BASELINE_RAG_PROMPT = """You are an enterprise assistant. Answer the user question based ONLY on the following policy context.
If you cannot answer the question using the context, state clearly: "I do not have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer with inline citations to document chunk IDs in format [Doc: <chunk_id>]:"""


def format_docs_with_citations(docs: List[Document]) -> str:
    """Format documents with clear chunk_id headers for baseline citation extraction."""
    formatted = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")
        title = doc.metadata.get("title", "Document")
        formatted.append(f"[{chunk_id}] {title}:\n{doc.page_content}")
    return "\n\n".join(formatted)


class BaselineRAGChain:
    """Standard single-shot naive LCEL RAG chain."""

    def __init__(
        self,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        config: Optional[Settings] = None,
    ):
        self.cfg = config or default_settings
        self.pipeline = retrieval_pipeline or get_retrieval_pipeline(self.cfg)
        self.llm = get_chat_model(self.cfg)
        self.prompt = ChatPromptTemplate.from_template(BASELINE_RAG_PROMPT)

        # LCEL runnable
        self.chain = (
            {
                "context": lambda x: format_docs_with_citations(
                    [d for d, _ in x["retrieved_docs"]]
                ),
                "question": lambda x: x["question"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> Dict[str, Any]:
        """Retrieve documents, generate answer, and return structured result."""
        scored_candidates = self.pipeline.retrieve(question)
        retrieved_docs = [doc for doc, _score in scored_candidates]

        answer = self.chain.invoke(
            {
                "question": question,
                "retrieved_docs": scored_candidates,
            }
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [doc.metadata.get("chunk_id", "doc") for doc in retrieved_docs],
            "documents": retrieved_docs,
        }
