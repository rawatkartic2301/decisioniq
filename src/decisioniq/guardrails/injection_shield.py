"""Defensive prompt injection shields and untrusted context isolation."""

import re
from typing import List

from langchain_core.documents import Document

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|directives)",
    r"system\s+override",
    r"you\s+are\s+now\s+in\s+debug\s+mode",
    r"disregard\s+(all\s+)?(guidelines|rules|guardrails)",
    r"reveal\s+(all\s+)?(passwords|credentials|ssn|salaries)",
    r"drop\s+table",
    r"delete\s+from",
]


def contains_injection_signature(text: str) -> bool:
    """Detect common prompt-injection and jailbreak attack signatures."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def sanitize_input(text: str) -> str:
    """Sanitize user input to neutralize XML/tag escaping attempts."""
    cleaned = text.replace("<untrusted_", "&lt;untrusted_").replace(
        "</untrusted_", "&lt;/untrusted_"
    )
    return cleaned.strip()


def wrap_untrusted_context(docs: List[Document]) -> str:
    """Isolate retrieved document text in delimited XML blocks with explicit untrusted tags."""
    formatted_blocks = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "chunk_unknown")
        title = doc.metadata.get("title", doc.metadata.get("filename", "Document"))
        dept = doc.metadata.get("department", "General")
        status = doc.metadata.get("status", "active")

        # Strip any internal delimiter injection attempts within chunk text
        clean_text = doc.page_content.replace("<untrusted_document_context>", "").replace(
            "</untrusted_document_context>", ""
        )

        block = (
            f'<untrusted_document_context id="{cid}" title="{title}" department="{dept}" status="{status}">\n'
            f"{clean_text.strip()}\n"
            f"</untrusted_document_context>"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)
