"""Hugging Face model and dataset discovery.

Uses the public Hugging Face Hub API — no API key required for read-only
search. Results are treated as discovery signals (model cards and dataset
pages are primary sources, but the agent should still verify claims against
the associated paper or technical report).
"""

import requests
from typing import List, Optional

from .config import AgentConfig, SOURCE_TYPE_GITHUB
from .models import SourceRecord

# Hugging Face source type
SOURCE_TYPE_HUGGINGFACE = "huggingface"


def _hf_source(
    title: str,
    url: str,
    metadata: Optional[dict] = None,
) -> SourceRecord:
    """Create a SourceRecord for a Hugging Face model or dataset."""
    meta = dict(metadata or {})
    meta["discovery_signal_only"] = False
    meta["requires_primary_source"] = True  # still verify against paper/report
    return SourceRecord(
        title=title,
        url=url,
        publication_date=None,
        source_type=SOURCE_TYPE_HUGGINGFACE,
        discovered_by="Implementation Scout",
        metadata=meta,
    )


def discover_huggingface_models(
    query: str,
    max_results: int = 10,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Search Hugging Face Hub for models matching a query.

    Uses the public /api/models endpoint. No authentication required.
    """
    cfg = config or AgentConfig()
    url = "https://huggingface.co/api/models"
    params = {
        "search": query,
        "limit": min(max_results, 20),
        "sort": "downloads",
        "direction": -1,
        "full": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: List[SourceRecord] = []
    for item in data:
        model_id = item.get("id", "")
        if not model_id:
            continue
        results.append(
            _hf_source(
                title=model_id,
                url=f"https://huggingface.co/{model_id}",
                metadata={
                    "author": item.get("author", ""),
                    "pipeline_tag": item.get("pipeline_tag", ""),
                    "task_categories": item.get("task_categories", []),
                    "tags": item.get("tags", []),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "last_modified": item.get("lastModified", ""),
                },
            )
        )
    return results


def discover_huggingface_datasets(
    query: str,
    max_results: int = 10,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Search Hugging Face Hub for datasets matching a query.

    Uses the public /api/datasets endpoint. No authentication required.
    """
    cfg = config or AgentConfig()
    url = "https://huggingface.co/api/datasets"
    params = {
        "search": query,
        "limit": min(max_results, 20),
        "sort": "downloads",
        "direction": -1,
        "full": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: List[SourceRecord] = []
    for item in data:
        dataset_id = item.get("id", "")
        if not dataset_id:
            continue
        results.append(
            _hf_source(
                title=dataset_id,
                url=f"https://huggingface.co/datasets/{dataset_id}",
                metadata={
                    "author": item.get("author", ""),
                    "task_categories": item.get("task_categories", []),
                    "tags": item.get("tags", []),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "last_modified": item.get("lastModified", ""),
                },
            )
        )
    return results


def discover_all_huggingface(
    topics: List[str],
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Discover Hugging Face models and datasets for all topics."""
    cfg = config or AgentConfig()
    all_sources: List[SourceRecord] = []

    for topic in topics:
        try:
            all_sources.extend(
                discover_huggingface_models(
                    topic, max_results=8, config=cfg
                )
            )
        except Exception as exc:
            all_sources.append(
                SourceRecord(
                    title=f"[ERROR] Hugging Face model search failed for: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_HUGGINGFACE,
                    discovered_by="Implementation Scout",
                    metadata={"error": str(exc)},
                )
            )

        try:
            all_sources.extend(
                discover_huggingface_datasets(
                    topic, max_results=5, config=cfg
                )
            )
        except Exception as exc:
            all_sources.append(
                SourceRecord(
                    title=f"[ERROR] Hugging Face dataset search failed for: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_HUGGINGFACE,
                    discovered_by="Implementation Scout",
                    metadata={"error": str(exc)},
                )
            )

    # Deduplicate by URL
    seen_urls = set()
    deduped: List[SourceRecord] = []
    for s in all_sources:
        if s.url and s.url in seen_urls:
            continue
        seen_urls.add(s.url)
        deduped.append(s)

    return deduped