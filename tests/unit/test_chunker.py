"""Unit tests for document chunker and deduplication."""

from langchain_core.documents import Document

from decisioniq.ingestion.chunker import DocumentChunker, clean_text, compute_sha256
from decisioniq.ingestion.metadata import deduplicate_chunks


def test_clean_text():
    raw = "Header\r\n\r\nText   with    spaces.\n\n\n\nExtra lines."
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunker_enriches_metadata():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    doc = Document(
        page_content="DecisionIQ is an enterprise agentic assistant. " * 10,
        metadata={"doc_id": "test_doc", "department": "Engineering"},
    )
    chunks = chunker.chunk_documents([doc])

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["doc_id"] == "test_doc"
        assert chunk.metadata["chunk_id"] == f"test_doc#chunk_{idx}"
        assert chunk.metadata["chunk_index"] == idx
        assert "content_hash" in chunk.metadata
        assert chunk.metadata["content_hash"] == compute_sha256(chunk.page_content)


def test_deduplicate_chunks():
    doc1 = Document(page_content="Duplicate chunk text", metadata={"content_hash": "hash123"})
    doc2 = Document(page_content="Duplicate chunk text", metadata={"content_hash": "hash123"})
    doc3 = Document(page_content="Unique chunk text", metadata={"content_hash": "hash456"})

    unique, dup_count = deduplicate_chunks([doc1, doc2, doc3])
    assert len(unique) == 2
    assert dup_count == 1
    assert [d.metadata["content_hash"] for d in unique] == ["hash123", "hash456"]
