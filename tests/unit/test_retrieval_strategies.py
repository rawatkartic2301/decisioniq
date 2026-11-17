"""Unit tests for retrieval strategies, BM25, RRF fusion, and baseline RAG chain."""

from langchain_core.documents import Document

from decisioniq.llm.mock import DeterministicEmbeddings, MockChatModel
from decisioniq.retrieval.baseline_rag import BaselineRAGChain, format_docs_with_citations
from decisioniq.retrieval.bm25 import BM25Retriever
from decisioniq.retrieval.filters import active_version_filter, build_metadata_filter
from decisioniq.retrieval.hybrid import reciprocal_rank_fusion
from decisioniq.retrieval.reranker import CrossEncoderReranker
from decisioniq.vectorstores.faiss_store import FAISSStore


def sample_docs():
    return [
        Document(
            page_content="Overtime for Grade L3 employees is paid at 1.5x regular pay.",
            metadata={"chunk_id": "ot#1", "department": "Operations", "status": "active"},
        ),
        Document(
            page_content="Carry forward leave policy allows up to 15 days into 2024.",
            metadata={"chunk_id": "leave#1", "department": "HR", "status": "active"},
        ),
        Document(
            page_content="Old 2022 policy allowed only 10 days carry forward leave.",
            metadata={"chunk_id": "leave#old", "department": "HR", "status": "deprecated"},
        ),
    ]


def test_bm25_retriever_keyword_search():
    docs = sample_docs()
    retriever = BM25Retriever(docs)

    results = retriever.retrieve("overtime L3", k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert top_doc.metadata["chunk_id"] == "ot#1"
    assert 0.0 <= score <= 1.0


def test_reciprocal_rank_fusion_math():
    doc_a = Document(page_content="Doc A", metadata={"chunk_id": "doc_a"})
    doc_b = Document(page_content="Doc B", metadata={"chunk_id": "doc_b"})

    # List 1: A at rank 1, B at rank 2
    list1 = [(doc_a, 0.9), (doc_b, 0.7)]
    # List 2: A at rank 2, B at rank 1
    # When rrf_k = 60:
    # doc_a gets 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # doc_b gets 1/(60+2) + 1/(60+1) = same
    # But if doc_a is rank 1 in both lists:
    list2 = [(doc_a, 0.95), (doc_b, 0.4)]
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=2)

    assert len(fused) == 2
    # doc_a should strictly be first
    assert fused[0][0].metadata["chunk_id"] == "doc_a"
    assert fused[0][1] > fused[1][1]


def test_metadata_filters():
    active_doc = Document(page_content="Active", metadata={"status": "active"})
    deprecated_doc = Document(page_content="Deprecated", metadata={"status": "deprecated"})

    assert active_version_filter(active_doc) is True
    assert active_version_filter(deprecated_doc) is False

    filter_dict = build_metadata_filter(active_only=True, department="Finance")
    assert filter_dict == {"status": "active", "department": "Finance"}


def test_cross_encoder_reranker_fallback():
    # Test fallback reranking without neural model download
    reranker = CrossEncoderReranker(enabled=False)
    query = "overtime rule L3"
    candidates = [
        (Document(page_content="Unrelated vacation notice", metadata={"chunk_id": "c1"}), 0.8),
        (
            Document(
                page_content="Grade L3 overtime rule is 1.5x pay", metadata={"chunk_id": "c2"}
            ),
            0.5,
        ),
    ]
    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    # The chunk with lexical overlap 'overtime', 'rule', 'l3' should be boosted to top
    assert reranked[0][0].metadata["chunk_id"] == "c2"


def test_format_docs_with_citations():
    docs = [
        Document(page_content="Text chunk 1", metadata={"chunk_id": "doc1#0", "title": "Doc One"}),
        Document(page_content="Text chunk 2", metadata={"chunk_id": "doc2#0", "title": "Doc Two"}),
    ]
    formatted = format_docs_with_citations(docs)
    assert "[doc1#0] Doc One:" in formatted
    assert "[doc2#0] Doc Two:" in formatted


def test_baseline_rag_chain_invoke(tmp_path):
    # Setup temporary store
    embeds = DeterministicEmbeddings(dimension=384)
    store = FAISSStore(embeddings=embeds, persist_path=tmp_path / "faiss_test")
    store.add_documents(sample_docs())

    from decisioniq.retrieval import RetrievalPipeline

    pipeline = RetrievalPipeline(vector_store=store, documents=sample_docs(), strategy="dense")
    rag = BaselineRAGChain(retrieval_pipeline=pipeline)
    # Monkeypatch to ensure deterministic mock LLM response
    rag.llm = MockChatModel(canned_response="Overtime for L3 is 1.5x [Doc: ot#1].")

    res = rag.invoke("What is overtime for L3?")
    assert "answer" in res
    assert "1.5x" in res["answer"]
    assert len(res["retrieved_chunks"]) > 0
