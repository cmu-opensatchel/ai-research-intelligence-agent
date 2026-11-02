"""Tests for chunking and metadata preservation."""

import pytest

from ai_research_agent.models import SourceRecord
from ai_research_agent.parser import chunk_text, clean_text, document_to_passages, extract_sections


def test_clean_text_normalizes_whitespace():
    assert clean_text("hello   world") == "hello world"
    assert clean_text("  leading and trailing  ") == "leading and trailing"


def test_chunk_text_preserves_content():
    text = "This is a test. " * 100  # long text
    chunks = chunk_text(text, max_chunk_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    # Verify no content is lost (all words appear across chunks)
    all_text = " ".join(chunks)
    for word in ["This", "is", "a", "test"]:
        assert word in all_text


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("Short text.", max_chunk_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_extract_sections_returns_general_when_no_headings():
    sections = extract_sections("Just some plain text with no headings at all.")
    assert len(sections) == 1
    assert sections[0]["section"] == "General"


def test_document_to_passages_preserves_metadata():
    source = SourceRecord(
        title="Test Paper",
        url="https://example.com/paper",
        publication_date="2024-01-15",
        source_type="arxiv",
        discovered_by="Research Scout",
        metadata={"authors": ["Alice", "Bob"]},
    )
    full_text = "Section One\nThis is the content of section one. " * 10
    passages = document_to_passages(source, full_text, max_chunk_chars=200, overlap_chars=20)

    assert len(passages) > 0
    for p in passages:
        assert p.source_url == "https://example.com/paper"
        assert p.source_title == "Test Paper"
        assert p.metadata["source_type"] == "arxiv"
        assert p.metadata["publication_date"] == "2024-01-15"
        assert p.metadata["discovered_by"] == "Research Scout"
        assert isinstance(p.chunk_index, int)