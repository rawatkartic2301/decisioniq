"""Document chunking and text cleaning module."""

import hashlib
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentChunker:
    """Splits documents into enriched chunks with unique chunk IDs and content hashes."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks and enrich metadata."""
        chunks: List[Document] = []

        for doc in documents:
            cleaned_content = clean_text(doc.page_content)
            if not cleaned_content:
                continue

            raw_chunks = self.splitter.split_text(cleaned_content)
            doc_id = doc.metadata.get("doc_id", "doc")

            for idx, text in enumerate(raw_chunks):
                chunk_meta = dict(doc.metadata)
                chunk_id = f"{doc_id}#chunk_{idx}"
                content_hash = compute_sha256(text)

                chunk_meta["chunk_id"] = chunk_id
                chunk_meta["chunk_index"] = idx
                chunk_meta["total_chunks"] = len(raw_chunks)
                chunk_meta["content_hash"] = content_hash

                chunks.append(Document(page_content=text, metadata=chunk_meta))

        return chunks
