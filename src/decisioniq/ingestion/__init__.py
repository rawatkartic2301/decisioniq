"""Ingestion pipeline components."""

from decisioniq.ingestion.chunker import DocumentChunker
from decisioniq.ingestion.loaders import load_directory, load_file
from decisioniq.ingestion.metadata import deduplicate_chunks
from decisioniq.ingestion.pipeline import run_ingestion_pipeline

__all__ = [
    "load_file",
    "load_directory",
    "DocumentChunker",
    "deduplicate_chunks",
    "run_ingestion_pipeline",
]
