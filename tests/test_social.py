"""Tests for optional social media discovery adapters.

Social media results are discovery signals only — never primary evidence.
Tests use mocks; no real API calls are made.
"""

from unittest.mock import patch

from ai_research_agent.config import AgentConfig
from ai_research_agent.models import SourceRecord
from ai_research_agent.social_discovery import (
    discover_all_social,
    discover_facebook,
    discover_instagram,
    discover_linkedin,
    discover_x,
    discover_youtube,
    SOURCE_TYPE_FACEBOOK,
    SOURCE_TYPE_INSTAGRAM,
    SOURCE_TYPE_LINKEDIN,
    SOURCE_TYPE_X,
    SOURCE_TYPE_YOUTUBE,
)


def test_youtube_no_api_key_returns_empty():
    """Without a YouTube API key, discovery returns empty list."""
    results = discover_youtube("agentic AI", config=AgentConfig(youtube_api_key=""))
    assert results == []


def test_youtube_disabled_by_default():
    """Social discovery is disabled by default; discover_all_social returns empty."""
    results = discover_all_social(["agentic AI"], config=AgentConfig())
    assert results == []


def test_x_no_token_returns_empty():
    """Without an X bearer token, discovery returns empty list."""
    results = discover_x("agentic AI", config=AgentConfig(x_bearer_token=""))
    assert results == []


def test_facebook_not_implemented():
    """Facebook returns a not-implemented placeholder, not real data."""
    results = discover_facebook("agentic AI")
    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_FACEBOOK
    assert "NOT IMPLEMENTED" in results[0].title
    assert results[0].metadata.get("discovery_signal_only") is True
    assert results[0].metadata.get("requires_primary_source") is True


def test_instagram_not_implemented():
    """Instagram returns a not-implemented placeholder."""
    results = discover_instagram("agentic AI")
    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_INSTAGRAM
    assert "NOT IMPLEMENTED" in results[0].title


def test_linkedin_not_implemented():
    """LinkedIn returns a not-implemented placeholder."""
    results = discover_linkedin("agentic AI")
    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_LINKEDIN
    assert "NOT IMPLEMENTED" in results[0].title


@patch("ai_research_agent.social_discovery.requests.get")
def test_youtube_mocked_api_response(mock_get):
    """YouTube adapter parses a mocked API response correctly."""
    mock_resp = mock_get.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "items": [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "AI Research Update",
                    "channelTitle": "AI Lab",
                    "publishedAt": "2026-01-15T10:00:00Z",
                    "description": "New developments in agentic AI",
                },
            }
        ]
    }

    results = discover_youtube(
        "agentic AI",
        api_key="fake-key",
        config=AgentConfig(youtube_api_key="fake-key"),
    )

    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_YOUTUBE
    assert results[0].title == "AI Research Update"
    assert "youtube.com/watch?v=abc123" in results[0].url
    assert results[0].metadata["channel"] == "AI Lab"
    assert results[0].metadata["discovery_signal_only"] is True


@patch("ai_research_agent.social_discovery.requests.get")
def test_x_mocked_api_response(mock_get):
    """X adapter parses a mocked API response correctly."""
    mock_resp = mock_get.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "data": [
            {
                "id": "1234567890",
                "text": "Exciting new paper on agentic AI just dropped!",
                "author_id": "user1",
                "created_at": "2026-01-15T10:00:00Z",
                "public_metrics": {"like_count": 10, "retweet_count": 5},
            }
        ]
    }

    results = discover_x(
        "agentic AI",
        bearer_token="fake-token",
        config=AgentConfig(x_bearer_token="fake-token"),
    )

    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_X
    assert "agentic AI" in results[0].title
    assert "twitter.com/i/web/status/1234567890" in results[0].url
    assert results[0].metadata["discovery_signal_only"] is True


def test_discover_all_social_disabled():
    """discover_all_social returns empty when social discovery is disabled."""
    results = discover_all_social(["agentic AI"], config=AgentConfig(enable_social_discovery=False))
    assert results == []


def test_social_sources_flagged_as_discovery_signals():
    """All social source records are flagged as discovery_signal_only."""
    results = discover_facebook("test") + discover_instagram("test") + discover_linkedin("test")
    for r in results:
        assert r.metadata.get("discovery_signal_only") is True
        assert r.metadata.get("requires_primary_source") is True
        assert isinstance(r, SourceRecord)