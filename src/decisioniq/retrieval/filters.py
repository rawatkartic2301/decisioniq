"""Metadata filtering predicates and post-retrieval filters."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document


def active_version_filter(doc: Document) -> bool:
    """Predicate: only include documents with status == 'active'."""
    return doc.metadata.get("status") == "active"


def build_metadata_filter(
    active_only: bool = True,
    department: Optional[str] = None,
    min_effective_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a dictionary filter for vector store similarity searches."""
    filters: Dict[str, Any] = {}
    if active_only:
        filters["status"] = "active"
    if department:
        filters["department"] = department
    if min_effective_date:
        filters["min_effective_date"] = min_effective_date
    return filters


def apply_post_retrieval_filters(
    results: List[Tuple[Document, float]],
    filter_func: Optional[Callable[[Document], bool]] = None,
) -> List[Tuple[Document, float]]:
    """Filter retrieved results using custom Python predicates."""
    if not filter_func:
        return results
    return [(doc, score) for doc, score in results if filter_func(doc)]
