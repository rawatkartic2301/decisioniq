"""Metadata enrichment and deduplication utilities."""

import logging
from typing import List, Set, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def deduplicate_chunks(
    chunks: List[Document],
    seen_hashes: Set[str] | None = None,
) -> Tuple[List[Document], int]:
    """Deduplicate chunks based on their SHA-256 content hashes.

    Returns:
        Tuple of (unique_chunks, duplicate_count)
    """
    seen = seen_hashes if seen_hashes is not None else set()
    unique_chunks: List[Document] = []
    duplicate_count = 0

    for chunk in chunks:
        chash = chunk.metadata.get("content_hash")
        if chash and chash in seen:
            duplicate_count += 1
            logger.debug("Skipping duplicate chunk hash: %s", chash)
            continue

        if chash:
            seen.add(chash)
        unique_chunks.append(chunk)

    logger.info(
        "Deduplication complete: %d unique chunks retained, %d duplicates removed",
        len(unique_chunks),
        duplicate_count,
    )
    return unique_chunks, duplicate_count
