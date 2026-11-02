"""Optional social media discovery adapters.

IMPORTANT SAFETY NOTES:
- Social media posts are DISCOVERY SIGNALS ONLY. They are never treated as
  technical proof or primary evidence. Any claim surfaced from social media
  must be backed by a primary source (arXiv paper, lab announcement, technical
  report, etc.) before inclusion in the roundup.
- No credentials are stored in this repository. API keys/tokens must be
  supplied at runtime via environment variables or config.
- Only official public APIs are used. No scraping that would violate a
  platform's Terms of Service.
- Facebook, Instagram, and LinkedIn are NOT implemented because they require
  OAuth app review and do not offer simple public read-only search APIs.
  Placeholders return empty results with a clear note.
"""

import os
from typing import List, Optional

import requests

from .config import AgentConfig
from .models import SourceRecord

# Source type constants
SOURCE_TYPE_YOUTUBE = "youtube"
SOURCE_TYPE_X = "x"
SOURCE_TYPE_FACEBOOK = "facebook"
SOURCE_TYPE_INSTAGRAM = "instagram"
SOURCE_TYPE_LINKEDIN = "linkedin"


def _social_source(
    title: str,
    url: str,
    source_type: str,
    discovered_by: str = "Social Scout",
    metadata: Optional[dict] = None,
) -> SourceRecord:
    """Create a SourceRecord flagged as a social discovery signal."""
    meta = dict(metadata or {})
    meta["discovery_signal_only"] = True
    meta["requires_primary_source"] = True
    return SourceRecord(
        title=title,
        url=url,
        publication_date=None,
        source_type=source_type,
        discovered_by=discovered_by,
        metadata=meta,
    )


def discover_youtube(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Search YouTube via the official YouTube Data API v3."""
    cfg = config or AgentConfig()
    key = api_key or cfg.youtube_api_key or os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, cfg.social_max_results_per_platform),
        "key": key,
    }
    try:
        resp = requests.get(url, params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: List[SourceRecord] = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")
        results.append(
            _social_source(
                title=snippet.get("title", ""),
                url=f"https://www.youtube.com/watch?v={video_id}",
                source_type=SOURCE_TYPE_YOUTUBE,
                metadata={
                    "channel": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", ""),
                },
            )
        )
    return results


def discover_x(
    query: str,
    bearer_token: Optional[str] = None,
    max_results: int = 5,
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Search X (Twitter) via the official API v2."""
    cfg = config or AgentConfig()
    token = bearer_token or cfg.x_bearer_token or os.environ.get("X_BEARER_TOKEN", "")
    if not token:
        return []

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": query,
        "max_results": min(max_results, cfg.social_max_results_per_platform),
        "tweet.fields": "created_at,author_id,public_metrics",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=cfg.request_timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: List[SourceRecord] = []
    for tweet in data.get("data", []):
        results.append(
            _social_source(
                title=tweet.get("text", "")[:100],
                url=f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                source_type=SOURCE_TYPE_X,
                metadata={
                    "author_id": tweet.get("author_id", ""),
                    "created_at": tweet.get("created_at", ""),
                    "public_metrics": tweet.get("public_metrics", {}),
                },
            )
        )
    return results


def discover_facebook(query: str, max_results: int = 5) -> List[SourceRecord]:
    """Facebook is NOT implemented.

    Requires OAuth app review and a user access token; there is no simple
    public read-only search API. Returns a not-implemented placeholder.
    """
    return [
        _social_source(
            title=f"[NOT IMPLEMENTED] Facebook search for: {query}",
            url="",
            source_type=SOURCE_TYPE_FACEBOOK,
            metadata={
                "note": "Facebook requires OAuth app review; no public read-only search API available.",
                "discovery_signal_only": True,
                "requires_primary_source": True,
            },
        )
    ]


def discover_instagram(query: str, max_results: int = 5) -> List[SourceRecord]:
    """Instagram is NOT implemented.

    Requires OAuth app review; there is no simple public read-only search API.
    Returns a not-implemented placeholder.
    """
    return [
        _social_source(
            title=f"[NOT IMPLEMENTED] Instagram search for: {query}",
            url="",
            source_type=SOURCE_TYPE_INSTAGRAM,
            metadata={
                "note": "Instagram requires OAuth app review; no public read-only search API available.",
                "discovery_signal_only": True,
                "requires_primary_source": True,
            },
        )
    ]


def discover_linkedin(query: str, max_results: int = 5) -> List[SourceRecord]:
    """LinkedIn is NOT implemented.

    Requires OAuth app review and partner permissions; there is no simple
    public read-only search API. Returns a not-implemented placeholder.
    """
    return [
        _social_source(
            title=f"[NOT IMPLEMENTED] LinkedIn search for: {query}",
            url="",
            source_type=SOURCE_TYPE_LINKEDIN,
            metadata={
                "note": "LinkedIn requires OAuth app review; no public read-only search API available.",
                "discovery_signal_only": True,
                "requires_primary_source": True,
            },
        )
    ]


def discover_all_social(
    topics: List[str],
    config: Optional[AgentConfig] = None,
) -> List[SourceRecord]:
    """Run all enabled social media discovery adapters.

    Only runs if config.enable_social_discovery is True. YouTube and X require
    API credentials provided at runtime. Facebook, Instagram, LinkedIn return
    not-implemented placeholders.
    """
    cfg = config or AgentConfig()
    if not cfg.enable_social_discovery:
        return []

    all_sources: List[SourceRecord] = []
    for topic in topics:
        # YouTube (requires API key)
        try:
            all_sources.extend(
                discover_youtube(
                    topic,
                    max_results=cfg.social_max_results_per_platform,
                    config=cfg,
                )
            )
        except Exception as exc:
            all_sources.append(
                _social_source(
                    title=f"[ERROR] YouTube discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_YOUTUBE,
                    metadata={"error": str(exc)},
                )
            )

        # X/Twitter (requires bearer token)
        try:
            all_sources.extend(
                discover_x(
                    topic,
                    max_results=cfg.social_max_results_per_platform,
                    config=cfg,
                )
            )
        except Exception as exc:
            all_sources.append(
                _social_source(
                    title=f"[ERROR] X discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_X,
                    metadata={"error": str(exc)},
                )
            )

        # Facebook / Instagram / LinkedIn (placeholders)
        try:
            all_sources.extend(discover_facebook(topic, max_results=cfg.social_max_results_per_platform))
        except Exception as exc:
            all_sources.append(
                _social_source(
                    title=f"[ERROR] Facebook discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_FACEBOOK,
                    metadata={"error": str(exc)},
                )
            )

        try:
            all_sources.extend(discover_instagram(topic, max_results=cfg.social_max_results_per_platform))
        except Exception as exc:
            all_sources.append(
                _social_source(
                    title=f"[ERROR] Instagram discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_INSTAGRAM,
                    metadata={"error": str(exc)},
                )
            )

        try:
            all_sources.extend(discover_linkedin(topic, max_results=cfg.social_max_results_per_platform))
        except Exception as exc:
            all_sources.append(
                _social_source(
                    title=f"[ERROR] LinkedIn discovery failed for topic: {topic}",
                    url="",
                    source_type=SOURCE_TYPE_LINKEDIN,
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