"""Unit tests for FAISS and Chroma vector store adapters."""

from pathlib import Path

from langchain_core.documents import Document

from decisioniq.llm.mock import DeterministicEmbeddings
from decisioniq.vectorstores.chroma_store import ChromaStore
from decisioniq.vectorstores.faiss_store import FAISSStore


def sample_documents():
    return [
        Document(
            page_content="Employees in Grade L3 receive 1.5x overtime multiplier for weekends.",
            metadata={"chunk_id": "doc_ot#chunk_0", "department": "Operations", "version": 2},
        ),
        Document(
            page_content="Standard carry-forward leave allows up to 15 days into the new year.",
            metadata={"chunk_id": "doc_leave#chunk_0", "department": "HR", "version": 2},
        ),
        Document(
            page_content="Deprecated 2022 policy allowed only 10 days of carry forward leave.",
            metadata={"chunk_id": "doc_leave#chunk_1", "department": "HR", "version": 1},
        ),
    ]


def test_faiss_store_lifecycle(tmp_path: Path):
    embeddings = DeterministicEmbeddings(dimension=384)
    persist_dir = tmp_path / "faiss_test"
    store = FAISSStore(embeddings=embeddings, persist_path=persist_dir)

    docs = sample_documents()
    ids = store.add_documents(docs)
    assert len(ids) == 3
    assert store.count() == 3

    # Similarity search
    results = store.similarity_search("overtime multiplier L3", k=2)
    assert len(results) > 0
    assert any("overtime" in r.page_content.lower() for r in results)

    # Search with score
    scored = store.similarity_search_with_score("overtime multiplier L3", k=2)
    assert len(scored) > 0
    doc, score = scored[0]
    assert 0.0 <= score <= 1.0

    # Search with metadata filter (HR department only)
    filtered = store.similarity_search("leave policy", k=5, filter={"department": "HR"})
    assert len(filtered) > 0
    for r in filtered:
        assert r.metadata["department"] == "HR"

    # Reload from disk
    store2 = FAISSStore(embeddings=embeddings, persist_path=persist_dir)
    assert store2.count() == 3


def test_chroma_store_lifecycle(tmp_path: Path):
    embeddings = DeterministicEmbeddings(dimension=384)
    persist_dir = tmp_path / "chroma_test"
    store = ChromaStore(
        embeddings=embeddings,
        persist_path=persist_dir,
        collection_name="test_collection",
    )

    docs = sample_documents()
    ids = store.add_documents(docs)
    assert len(ids) == 3
    assert store.count() == 3

    # Similarity search
    results = store.similarity_search("overtime multiplier L3", k=2)
    assert len(results) > 0

    # Search with score
    scored = store.similarity_search_with_score("overtime multiplier L3", k=2)
    assert len(scored) > 0
    doc, score = scored[0]
    assert 0.0 <= score <= 1.0

    # Search with metadata filter (version 2 only)
    filtered = store.similarity_search("leave policy", k=5, filter={"version": 2})
    assert len(filtered) > 0
    for r in filtered:
        assert r.metadata["version"] == 2
