"""Source discovery from public research sources (arXiv, GitHub, lab announcements).

Keep adapters simple and replaceable. Each source record distinguishes the
publication date from the retrieval time.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Optional

from .config import (
    AgentConfig,
    REQUEST_TIMEOUT,
    SOURCE_TYPE_ARXIV,
    SOURCE_TYPE_GITHUB,
    SOURCE_TYPE_LAB_ANNOUNCEMENT,
    USER_AGENT,
)
from .models import SourceRecord


def _fetch_with_retry(url: str, timeout: int = REQUEST_TIMEOUT, max_retries: int = 2) -> Optional[requests.Response]:
    """Fetch a URL with bounded retries and a proper user agent."""
    headers = {"User-Agent": USER_AGENT}
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_error = exc
    return None


def discover_arxiv(topics: List[str], max_results_per_topic: int = 10, config: Optional[AgentConfig] = None) -> List[SourceRecord]:
    """Discover arXiv preprints matching the given topics."""
    sources: List[SourceRecord] = []
    cfg = config or AgentConfig()

    for topic in topics:
        query = f"all:{topic}"
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query={query}&start=0&max_results={max_results_per_topic}"
        )
        resp = _fetch_with_retry(url, timeout=cfg.request_timeout, max_retries=cfg.max_retries)
        if resp is None:
            continue

        try:
            feed = feedparser.parse(resp.text)
        except Exception:
            continue

        for entry in feed.entries:
            published = getattr(entry, "published", None)
            sources.append(
                SourceRecord(
                    title=getattr(entry, "title", "").strip(),
                    url=getattr(entry, "id", ""),
                    publication_date=published,
                    source_type=SOURCE_TYPE_ARXIV,
                    discovered_by="Research Scout",
                    metadata={
                        "authors": [a.name for a in getattr(entry, "authors", [])],
                        "summary": getattr(entry, "summary", ""),
                        "arxiv_id": getattr(entry, "id", "").split("/abs/")[-1] if "/abs/" in getattr(entry, "id", "") else "",
                    },
                )
            )

    return sources


def discover_github(topic: str, max_results: int = 10, config: Optional[AgentConfig] = None) -> List[SourceRecord]:
    """Discover GitHub repositories matching a topic via the GitHub Search API."""
    sources: List[SourceRecord] = []
    cfg = config or AgentConfig()

    url = "https://api.github.com/search/repositories"
    params = {"q": topic, "per_page": max_results, "sort": "updated"}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github.v3+json"}

    resp = _fetch_with_retry(url, timeout=cfg.request_timeout, max_retries=cfg.max_retries)
    if resp is None:
        return sources

    try:
        data = resp.json()
    except Exception:
        return sources

    for item in data.get("items", []):
        sources.append(
            SourceRecord(
                title=item.get("full_name", ""),
                url=item.get("html_url", ""),
                publication_date=item.get("pushed_at"),
                source_type=SOURCE_TYPE_GITHUB,
                discovered_by="Implementation Scout",
                metadata={
                    "description": item.get("description", ""),
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "language": item.get("language", ""),
                    "topics": item.get("topics", []),
                },
            )
        )

    return sources


def discover_lab_announcement(url: str, config: Optional[AgentConfig] = None) -> Optional[SourceRecord]:
    """Fetch a single lab announcement / technical report page and extract metadata."""
    cfg = config or AgentConfig()
    resp = _fetch_with_retry(url, timeout=cfg.request_timeout, max_retries=cfg.max_retries)
    if resp is None:
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return None

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    return SourceRecord(
        title=title,
        url=url,
        publication_date=None,
        source_type=SOURCE_TYPE_LAB_ANNOUNCEMENT,
        discovered_by="Industry Scout",
        metadata={
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_length": len(resp.text),
        },
    )


def discover_all_sources(topics: List[str], config: Optional[AgentConfig] = None) -> List[SourceRecord]:
    """Run all discovery adapters and return a deduplicated list of sources."""
    cfg = config or AgentConfig()
    all_sources: List[SourceRecord] = []

    for topic in topics:
        try:
            all_sources.extend(discover_arxiv([topic], max_results_per_topic=8, config=cfg))
        except Exception as exc:
            all_sources.append(
                SourceRecord(
                    title=f"[ERROR] arXiv discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_ARXIV,
                    discovered_by="Research Scout",
                    metadata={"error": str(exc)},
                )
            )

    for topic in topics:
        try:
            all_sources.extend(discover_github(topic, max_results=5, config=cfg))
        except Exception as exc:
            all_sources.append(
                SourceRecord(
                    title=f"[ERROR] GitHub discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_GITHUB,
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