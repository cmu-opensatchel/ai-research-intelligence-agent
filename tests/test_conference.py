"""Tests for conference discovery (mocked — no real API calls)."""

from unittest.mock import patch

from ai_research_agent.config import AgentConfig
from ai_research_agent.conference_discovery import (
    discover_all_conferences,
    discover_conference_papers,
    SOURCE_TYPE_CONFERENCE,
)
from ai_research_agent.models import SourceRecord


@patch("ai_research_agent.conference_discovery.requests.get")
def test_conference_papers_mocked(mock_get):
    """Conference adapter parses a mocked OpenReview response."""
    mock_resp = mock_get.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "id": "note123",
            "forum": "forum123",
            "content": {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "abstract": "We propose a new network architecture...",
            },
            "cdate": 1500000000000,
        }
    ]

    results = discover_conference_papers(
        ["transformer"], venues=["neurips"], config=AgentConfig()
    )

    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_CONFERENCE
    assert results[0].title == "Attention Is All You Need"
    assert "openreview.net/forum?id=forum123" in results[0].url
    assert results[0].metadata["venue"] == "NeurIPS"
    assert "Ashish Vaswani" in results[0].metadata["authors"]


@patch("ai_research_agent.conference_discovery.requests.get")
def test_conference_api_error_returns_empty(mock_get):
    """Conference adapter returns empty list on API error (no crash)."""
    mock_get.side_effect = Exception("Network error")
    results = discover_conference_papers(["transformer"], config=AgentConfig())
    assert results == []


def test_discover_all_conferences_venues():
    """discover_all_conferences searches across multiple venues."""
    with patch("ai_research_agent.conference_discovery.discover_conference_papers") as m:
        m.return_value = []
        results = discover_all_conferences(["bert"], config=AgentConfig())
        # Should have been called with default venues
        assert m.called
        call_args = m.call_args
        assert "venues" not in call_args[1] or call_args[1]["venues"] is None