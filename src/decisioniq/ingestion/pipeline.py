"""End-to-end ingestion pipeline with CLI entrypoint."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from decisioniq.config import settings
from decisioniq.ingestion.chunker import DocumentChunker
from decisioniq.ingestion.data_gen import generate_database, write_document_files
from decisioniq.ingestion.loaders import load_directory
from decisioniq.ingestion.metadata import deduplicate_chunks
from decisioniq.llm.factory import get_embeddings
from decisioniq.vectorstores import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    store_type: Literal["faiss", "chroma"] = "faiss",
    docs_dir: Path = Path("data/docs"),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force_data_gen: bool = False,
    persist_path: Optional[Path] = None,
) -> int:
    """Execute document loading, chunking, deduplication, and vector store indexing."""
    start_time = time.time()
    logger.info("Starting ingestion pipeline targeting store: %s", store_type)

    # 1. Ensure synthetic data exists
    if force_data_gen or not docs_dir.exists() or len(list(docs_dir.glob("*.md"))) < 5:
        logger.info("Generating synthetic documents and relational database...")
        write_document_files()
        generate_database()

    # 2. Load documents
    raw_docs = load_directory(docs_dir)
    if not raw_docs:
        logger.error("No valid documents found in %s", docs_dir)
        return 0

    # 3. Chunk documents
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(raw_docs)
    logger.info("Created %d chunks from %d source documents", len(chunks), len(raw_docs))

    # 4. Deduplicate
    unique_chunks, dup_count = deduplicate_chunks(chunks)

    # 5. Initialize vector store & embeddings
    embeddings = get_embeddings(settings)
    vector_store = get_vector_store(
        store_type=store_type,
        embeddings=embeddings,
        config=settings,
        persist_path=persist_path,
    )

    # 6. Index into vector store
    indexed_ids = vector_store.add_documents(unique_chunks)
    vector_store.persist()

    elapsed = time.time() - start_time
    logger.info(
        "Successfully indexed %d chunks into %s in %.2f seconds (total stored: %d)",
        len(indexed_ids),
        store_type.upper(),
        elapsed,
        vector_store.count(),
    )
    return len(indexed_ids)


def main():
    parser = argparse.ArgumentParser(description="DecisionIQ Document Ingestion Pipeline")
    parser.add_argument(
        "--store",
        choices=["faiss", "chroma"],
        default=settings.vector_store,
        help="Target vector store (faiss or chroma)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data/docs"),
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Recursive text chunk size",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Recursive text chunk overlap",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of synthetic data corpus",
    )

    args = parser.parse_args()
    count = run_ingestion_pipeline(
        store_type=args.store,
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_data_gen=args.regenerate_data,
    )
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
