"""Lab RSS feed discovery.

Parses RSS/Atom feeds from research-lab blogs and announcements. Users can
configure a list of feed URLs; the agent fetches and parses entries,
extracting title, URL, publication date, and a summary.

No API key required — just public RSS feeds.
"""

import feedparser
from typing import List, Optional

from .config import AgentConfig, SOURCE_TYPE_LAB_ANNOUNCEMENT
from .models import SourceRecord


def discover_rss_feed(
    feed_url: str,
    max_entries: int = 10,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Fetch and parse a single RSS/Atom feed.

    Returns SourceRecord objects with publication_date and retrieval_time
    clearly distinguished.
    """
    cfg = config or AgentConfig()
    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        return []

    results: List[SourceRecord] = []
    for entry in feed.entries[:max_entries]:
        title = getattr(entry, "title", "") or feed_url
        link = getattr(entry, "link", "")
        published = getattr(entry, "published", None) or getattr(
            entry, "updated", None
        )
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "")

        results.append(
            SourceRecord(
                title=title.strip(),
                url=link,
                publication_date=published,
                source_type=SOURCE_TYPE_LAB_ANNOUNCEMENT,
                discovered_by="Industry Scout",
                metadata={
                    "feed_url": feed_url,
                    "summary": summary,
                    "authors": [a.name for a in getattr(entry, "authors", [])],
                    "tags": [t.term for t in getattr(entry, "tags", [])],
                },
            )
        )

    return results


def discover_all_rss_feeds(
    feed_urls: List[str],
    max_entries_per_feed: int = 10,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Fetch and parse multiple RSS/Atom feeds."""
    cfg = config or AgentConfig()
    all_sources: List[SourceRecord] = []

    for feed_url in feed_urls:
        try:
            all_sources.extend(
                discover_rss_feed(
                    feed_url,
                    max_entries=max_entries_per_feed,
                    config=cfg,
                )
            )
        except Exception as exc:
            all_sources.append(
                SourceRecord(
                    title=f"[ERROR] RSS feed failed: {feed_url}",
                    url=feed_url,
                    source_type=SOURCE_TYPE_LAB_ANNOUNCEMENT,
                    discovered_by="Industry Scout",
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


# Default lab RSS feeds (configurable by user)
DEFAULT_LAB_RSS_FEEDS: List[str] = [
    # Add your lab/blog RSS feed URLs here
    # "https://deepmind.google/blog/rss.xml",
    # "https://ai.meta.com/blog/rss/",
    # "https://openai.com/blog/rss/",
    # "https://www.anthropic.com/news/rss",
    # "https://research.google/blog/rss/",
    # "https://www.microsoft.com/en-us/research/feed/",
]