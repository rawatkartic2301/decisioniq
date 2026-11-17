"""Integration test for full ingestion pipeline into both FAISS and Chroma."""

from pathlib import Path

from decisioniq.ingestion.pipeline import run_ingestion_pipeline
from decisioniq.llm.mock import DeterministicEmbeddings
from decisioniq.vectorstores.chroma_store import ChromaStore
from decisioniq.vectorstores.faiss_store import FAISSStore


def test_ingestion_pipeline_faiss(tmp_path: Path):
    faiss_dir = tmp_path / "faiss_out"
    count = run_ingestion_pipeline(
        store_type="faiss",
        docs_dir=Path("data/docs"),
        chunk_size=400,
        chunk_overlap=40,
        persist_path=faiss_dir,
    )
    assert count > 0

    # Verify reload
    embeds = DeterministicEmbeddings(dimension=384)
    store = FAISSStore(embeddings=embeds, persist_path=faiss_dir)
    assert store.count() == count

    results = store.similarity_search("overtime rule grade L3", k=3)
    assert len(results) > 0
    assert "chunk_id" in results[0].metadata


def test_ingestion_pipeline_chroma(tmp_path: Path):
    chroma_dir = tmp_path / "chroma_out"
    count = run_ingestion_pipeline(
        store_type="chroma",
        docs_dir=Path("data/docs"),
        chunk_size=400,
        chunk_overlap=40,
        persist_path=chroma_dir,
    )
    assert count > 0

    # Verify query
    embeds = DeterministicEmbeddings(dimension=384)
    store = ChromaStore(embeddings=embeds, persist_path=chroma_dir)
    assert store.count() == count

    results = store.similarity_search("leave carry forward", k=3)
    assert len(results) > 0
    assert "chunk_id" in results[0].metadata
