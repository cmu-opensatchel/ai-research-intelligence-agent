"""Tests for handling unavailable sources or malformed responses."""

import pytest

from ai_research_agent.discovery import _fetch_with_retry, discover_all_sources
from ai_research_agent.models import SourceRecord


def test_fetch_with_retry_invalid_url():
    """Fetching an invalid URL should return None, not crash."""
    result = _fetch_with_retry("http://invalid-host-that-does-not-exist.xyz/", timeout=5, max_retries=1)
    assert result is None


def test_discover_all_sources_handles_errors_gracefully():
    """Discovery should not crash when external sources are unavailable."""
    sources = discover_all_sources(["nonexistent-topic-xyz123"])
    # Should return a list (possibly empty or with error records), not crash
    assert isinstance(sources, list)


def test_error_source_record_created():
    """Error sources should be recorded as SourceRecord with error metadata."""
    source = SourceRecord(
        title="[ERROR] Test",
        url="",
        source_type="arxiv",
        discovered_by="Research Scout",
        metadata={"error": "Connection failed"},
    )
    assert source.metadata["error"] == "Connection failed"
    assert source.url == ""


def test_source_record_distinguishes_dates():
    """SourceRecord should distinguish publication_date from retrieval_time."""
    source = SourceRecord(
        title="Paper",
        url="https://arxiv.org/abs/2401.00001",
        publication_date="2024-01-15",
        source_type="arxiv",
        discovered_by="Research Scout",
    )
    assert source.publication_date == "2024-01-15"
    assert source.retrieval_time is not None
    assert source.retrieval_time != source.publication_date