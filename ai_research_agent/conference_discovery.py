"""Conference proceedings discovery via OpenReview.

OpenReview hosts conference proceedings for NeurIPS, ICML, ICLR, and many
other ML/AI conferences. The public API requires no authentication for
read-only search. Results are primary sources (conference papers).
"""

import requests
from typing import List, Optional

from .config import AgentConfig
from .models import SourceRecord

# Conference source type
SOURCE_TYPE_CONFERENCE = "conference"

# Known conference venues on OpenReview
CONFERENCE_VENUES = {
    "neurips": "NeurIPS",
    "icml": "ICML",
    "iclr": "ICLR",
    "aistats": "AISTATS",
    "kdd": "KDD",
}


def _conference_source(
    title: str,
    url: str,
    venue: str,
    metadata: Optional[dict] = None,
) -> SourceRecord:
    """Create a SourceRecord for a conference paper."""
    meta = dict(metadata or {})
    meta["discovery_signal_only"] = False
    meta["requires_primary_source"] = False  # conference papers are primary sources
    return SourceRecord(
        title=title,
        url=url,
        publication_date=None,
        source_type=SOURCE_TYPE_CONFERENCE,
        discovered_by="Research Scout",
        metadata=meta,
    )


def _fetch_openreview_notes(
    venue: str,
    query: str,
    limit: int = 20,
    config: Optional[AgentConfig] = None,
) -> List[dict]:
    """Fetch notes from OpenSearch/OpenReview API for a given venue."""
    cfg = config or AgentConfig()
    url = "https://api.openreview.net/notes"
    params = {
        "content.search": query,
        "limit": limit,
        "details": "invited,replyCount",
    }
    try:
        resp = requests.get(url, params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def discover_conference_papers(
    topics: List[str],
    venues: Optional[List[str]] = None,
    max_results_per_topic: int = 5,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Discover conference papers from OpenReview matching the given topics.

    venues: list of conference keys (neurips, icml, iclr, etc.). If None,
    searches across all venues.
    """
    cfg = config or AgentConfig()
    all_sources: List[SourceRecord] = []

    venue_list = venues or list(CONFERENCE_VENUES.keys())

    for topic in topics:
        for venue_key in venue_list:
            try:
                notes = _fetch_openreview_notes(
                    venue_key, topic, limit=max_results_per_topic, config=cfg
                )
                for note in notes:
                    content = note.get("content", {})
                    title = content.get("title", "")
                    if not title:
                        continue
                    forum_id = note.get("forum", "") or note.get("id", "")
                    url = f"https://openreview.net/forum?id={forum_id}"
                    results = content.get("authors", [])
                    all_sources.append(
                        _conference_source(
                            title=title,
                            url=url,
                            venue=CONFERENCE_VENUES.get(venue_key, venue_key),
                            metadata={
                                "venue": CONFERENCE_VENUES.get(venue_key, venue_key),
                                "authors": content.get("authors", []),
                                "abstract": content.get("abstract", ""),
                                "openreview_id": note.get("id", ""),
                                "submitted_date": note.get("cdate", ""),
                            },
                        )
                    )
            except Exception:
                continue

    # Deduplicate by URL
    seen_urls = set()
    deduped: List[SourceRecord] = []
    for s in all_sources:
        if s.url and s.url in seen_urls:
            continue
        seen_urls.add(s.url)
        deduped.append(s)

    return deduped


def discover_all_conferences(
    topics: List[str],
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Discover conference papers for all topics across known venues."""
    return discover_conference_papers(
        topics, venues=None, max_results_per_topic=5, config=config
    )