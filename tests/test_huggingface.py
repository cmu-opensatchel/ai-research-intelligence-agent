"""Tests for Hugging Face discovery (mocked — no real API calls)."""

from unittest.mock import patch

from ai_research_agent.config import AgentConfig
from ai_research_agent.huggingface_discovery import (
    discover_all_huggingface,
    discover_huggingface_datasets,
    discover_huggingface_models,
    SOURCE_TYPE_HUGGINGFACE,
)
from ai_research_agent.models import SourceRecord


@patch("ai_research_agent.huggingface_discovery.requests.get")
def test_huggingface_models_mocked(mock_get):
    """HF models adapter parses a mocked API response."""
    mock_resp = mock_get.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "id": "bert-base-uncased",
            "author": "google-bert",
            "pipeline_tag": "fill-mask",
            "task_categories": ["text-classification"],
            "tags": ["nlp", "bert"],
            "downloads": 1000000,
            "likes": 5000,
            "lastModified": "2024-01-01T00:00:00Z",
        }
    ]

    results = discover_huggingface_models(
        "bert", config=AgentConfig()
    )

    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_HUGGINGFACE
    assert results[0].title == "bert-base-uncased"
    assert "huggingface.co/bert-base-uncased" in results[0].url
    assert results[0].metadata["author"] == "google-bert"
    assert results[0].metadata["downloads"] == 1000000


@patch("ai_research_agent.huggingface_discovery.requests.get")
def test_huggingface_datasets_mocked(mock_get):
    """HF datasets adapter parses a mocked API response."""
    mock_resp = mock_get.return_value
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "id": "squad",
            "author": "stanfordnlp",
            "task_categories": ["question-answering"],
            "tags": ["nlp", "qa"],
            "downloads": 500000,
            "likes": 2000,
            "lastModified": "2024-01-01T00:00:00Z",
        }
    ]

    results = discover_huggingface_datasets(
        "squad", config=AgentConfig()
    )

    assert len(results) == 1
    assert results[0].source_type == SOURCE_TYPE_HUGGINGFACE
    assert results[0].title == "squad"
    assert "huggingface.co/datasets/squad" in results[0].url


@patch("ai_research_agent.huggingface_discovery.requests.get")
def test_huggingface_api_error_returns_empty(mock_get):
    """HF adapter returns empty list on API error (no crash)."""
    mock_get.side_effect = Exception("Network error")
    results = discover_huggingface_models("bert", config=AgentConfig())
    assert results == []


def test_discover_all_huggingface_dedupes():
    """discover_all_huggingface deduplicates by URL."""
    with patch("ai_research_agent.huggingface_discovery.discover_huggingface_models") as m1, \
         patch("ai_research_agent.huggingface_discovery.discover_huggingface_datasets") as m2:
        m1.return_value = [
            SourceRecord(title="Model A", url="https://huggingface.co/model-a", source_type="huggingface", discovered_by="Implementation Scout")
        ]
        m2.return_value = [
            SourceRecord(title="Model A", url="https://huggingface.co/model-a", source_type="huggingface", discovered_by="Implementation Scout")
        ]
        results = discover_all_huggingface(["bert"], config=AgentConfig())
        assert len(results) == 1  # deduplicated