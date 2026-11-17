"""Multi-format document loaders supporting MD, TXT, PDF, and DOCX with metadata."""

import json
import logging
from pathlib import Path
from typing import List, Optional

from docx import Document as DocxReader
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_file(file_path: Path) -> Optional[Document]:
    """Load an individual document file and return a LangChain Document with metadata."""
    if not file_path.exists():
        logger.warning("File does not exist: %s", file_path)
        return None

    ext = file_path.suffix.lower()
    content = ""

    try:
        if ext in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        elif ext == ".pdf":
            reader = PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            content = "\n\n".join(pages)

        elif ext == ".docx":
            doc = DocxReader(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n\n".join(paragraphs)

        else:
            logger.debug("Skipping unsupported file extension: %s", ext)
            return None

    except Exception as e:
        logger.error("Failed to load file %s: %s", file_path, e)
        return None

    # Load sidecar metadata if available
    metadata = {
        "source": str(file_path),
        "filename": file_path.name,
        "extension": ext,
        "doc_id": file_path.stem,
        "status": "active",
        "version": 1,
    }

    meta_file = file_path.parent / f"{file_path.name}.meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                sidecar_meta = json.load(mf)
                metadata.update(sidecar_meta)
        except Exception as e:
            logger.warning("Failed to parse sidecar metadata %s: %s", meta_file, e)

    return Document(page_content=content, metadata=metadata)


def load_directory(directory_path: Path) -> List[Document]:
    """Recursively load all supported documents from a directory."""
    documents: List[Document] = []
    if not directory_path.exists():
        logger.warning("Directory does not exist: %s", directory_path)
        return documents

    for path in directory_path.iterdir():
        if path.is_file() and not path.name.endswith(".meta.json"):
            doc = load_file(path)
            if doc and doc.page_content.strip():
                documents.append(doc)

    logger.info("Loaded %d documents from %s", len(documents), directory_path)
    return documents
