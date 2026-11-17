"""PII (Personally Identifiable Information) scrubber for logs, audits, and traces."""

import re

# Regex patterns for common sensitive enterprise PII
PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{20,}|key-[a-zA-Z0-9]{20,}|decisioniq_[a-zA-Z0-9_]{10,})\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


def scrub_pii(text: str) -> str:
    """Mask sensitive PII identifiers (SSN, credit cards, secret keys, phones) with redaction markers."""
    if not text:
        return text

    scrubbed = text
    for pii_type, pattern in PATTERNS.items():
        scrubbed = re.sub(pattern, f"[REDACTED_{pii_type}]", scrubbed)

    return scrubbed
