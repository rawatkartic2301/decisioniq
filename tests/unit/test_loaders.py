"""Unit tests for document loaders."""

from pathlib import Path

from decisioniq.ingestion.loaders import load_directory, load_file


def test_load_markdown_and_metadata(tmp_path: Path):
    md_file = tmp_path / "test_policy.md"
    meta_file = tmp_path / "test_policy.md.meta.json"

    md_file.write_text("# Test Policy\nThis is a sample corporate policy.", encoding="utf-8")
    meta_file.write_text(
        '{"doc_id": "test_001", "version": 2, "department": "HR"}', encoding="utf-8"
    )

    doc = load_file(md_file)
    assert doc is not None
    assert "sample corporate policy" in doc.page_content
    assert doc.metadata["doc_id"] == "test_001"
    assert doc.metadata["version"] == 2
    assert doc.metadata["department"] == "HR"


def test_load_directory_filters_metadata_json(tmp_path: Path):
    (tmp_path / "doc1.txt").write_text("Hello text 1", encoding="utf-8")
    (tmp_path / "doc2.md").write_text("Hello markdown 2", encoding="utf-8")
    (tmp_path / "doc2.md.meta.json").write_text('{"title": "Meta"}', encoding="utf-8")

    docs = load_directory(tmp_path)
    # Should only load doc1.txt and doc2.md, ignoring .meta.json as standalone doc
    assert len(docs) == 2
    filenames = {d.metadata["filename"] for d in docs}
    assert filenames == {"doc1.txt", "doc2.md"}
