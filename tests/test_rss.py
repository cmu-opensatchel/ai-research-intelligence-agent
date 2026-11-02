"""Tests for RSS feed discovery (mocked — no real HTTP calls)."""

from types import SimpleNamespace
from unittest.mock import patch

from ai_research_agent.config import AgentConfig
from ai_research_agent.rss_discovery import (
    discover_all_rss_feeds,
    discover_rss_feed,
    SOURCE_TYPE_LAB_ANNOUNCEMENT,
)


@patch("ai_research_agent.rss_discovery.feedparser.parse")
def test_rss_feed_mocked(mock_parse):
    """RSS adapter parses a mocked feed correctly."""
    mock_feed = mock_parse.return_value
    mock_feed.entries = [
        SimpleNamespace(
            title="New Research Release",
            link="https://example.com/blog/new-release",
            published="2026-01-15T10:00:00Z",
            summary="We just published a new paper on agentic AI.",
            authors=[SimpleNamespace(name="Jane Doe")],
            tags=[SimpleNamespace(term="AI"), SimpleNamespace(term="research")],
        )
    ]

    results = discover_rss_feed(
        "https://example.com/rss", config=AgentConfig()
    )

    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_LAB_ANNOUNCEMENT
    assert results[0].title == "New Research Release"
    assert results[0].url == "https://example.com/blog/new-release"
    assert results[0].publication_date == "2026-01-15T10:00:00Z"
    assert results[0].metadata["feed_url"] == "https://example.com/rss"


@patch("ai_research_agent.rss_discovery.feedparser.parse")
def test_rss_feed_error_returns_empty(mock_parse):
    """RSS adapter returns empty list on parse error (no crash)."""
    mock_parse.side_effect = Exception("Invalid feed")
    results = discover_rss_feed("https://example.com/rss", config=AgentConfig())
    assert results == []


@patch("ai_research_agent.rss_discovery.discover_rss_feed")
def test_discover_all_rss_feeds_calls_each_feed(mock_fetch):
    """discover_all_rss_feeds calls discover_rss_feed for each feed URL."""
    mock_fetch.return_value = []
    results = discover_all_rss_feeds(
        ["https://feed1.com/rss", "https://feed2.com/rss"], config=AgentConfig()
    )
    assert mock_fetch.call_count == 2
    assert results == []