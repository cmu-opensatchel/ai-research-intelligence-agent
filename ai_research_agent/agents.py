"""Agent role implementations.

Logical agent roles are Research Scout, Industry Scout, Implementation Scout,
Impact Analyst, and Research Coordinator. Implemented as separate functions
that can be composed by the coordinator. No CrewAI, LangChain, or MCP is used.
"""

from typing import List, Optional

from .config import AgentConfig
from .discovery import discover_all_sources
from .models import Candidate, SourceRecord
from .parser import document_to_passages
from .retrieval import Retriever
from .synthesis import deduplicate_candidates
from .tot import ToTAssessor
from .verification import EvidenceVerifier, verify_claims_for_candidates


def research_scout(topics: List[str], config: Optional[AgentConfig] = None) -> List[SourceRecord]:
    """Research Scout: discovers arXiv preprints and research papers."""
    cfg = config or AgentConfig()
    return discover_all_sources(topics, config=cfg)


def industry_scout(announcement_urls: List[str], config: Optional[AgentConfig] = None) -> List[SourceRecord]:
    """Industry Scout: fetches lab announcements and technical reports."""
    cfg = config or AgentConfig()
    sources: List[SourceRecord] = []
    for url in announcement_urls:
        from .discovery import discover_lab_announcement
        src = discover_lab_announcement(url, config=cfg)
        if src:
            sources.append(src)
    return sources


def implementation_scout(topics: List[str], config: Optional[AgentConfig] = None) -> List[SourceRecord]:
    """Implementation Scout: discovers GitHub repositories and implementations."""
    cfg = config or AgentConfig()
    from .discovery import discover_github
    sources: List[SourceRecord] = []
    for topic in topics:
        sources.extend(discover_github(topic, max_results=5, config=cfg))
    return sources


def social_scout(
    topics: List[str], config: Optional[AgentConfig] = None
) -> List[SourceRecord]:
    """Social Scout: discovers social media signals (YouTube, X, etc.).

    Only runs if config.enable_social_discovery is True. Results are discovery
    signals only — never primary evidence.
    """
    cfg = config or AgentConfig()
    if not cfg.enable_social_discovery:
        return []

    from .social_discovery import discover_all_social
    return discover_all_social(topics, config=cfg)


def impact_analyst(
    candidates: List[Candidate],
    passages: List,
    verification_results: List,
    config: Optional[AgentConfig] = None,
) -> dict:
    """Impact Analyst: runs ToT assessment and ranks candidates."""
    cfg = config or AgentConfig()
    assessor = ToTAssessor(config=cfg)
    candidate_branches: dict = {}
    for candidate in candidates:
        candidate_passages = [p for p in passages if p.source_url == candidate.source.url]
        branches = assessor.assess_candidate(candidate, candidate_passages, verification_results)
        candidate_branches[candidate.id] = branches

    from .tot import rank_candidates
    rankings = rank_candidates(candidate_branches, config=cfg)
    return {"candidate_branches": candidate_branches, "rankings": rankings}


def research_coordinator(
    topics: List[str],
    reporting_window_days: int,
    announcement_urls: Optional[List[str]] = None,
    config: Optional[AgentConfig] = None,
) -> dict:
    """Research Coordinator: orchestrates the full workflow end-to-end."""
    cfg = config or AgentConfig()
    cfg.topics = topics
    cfg.reporting_window_days = reporting_window_days

    errors: List[str] = []

    # Step 1: Discovery
    research_sources = research_scout(topics, config=cfg)
    implementation_sources = implementation_scout(topics, config=cfg)
    industry_sources = industry_scout(announcement_urls or [], config=cfg)
    social_sources = social_scout(topics, config=cfg)

    all_sources = (
        research_sources + implementation_sources + industry_sources + social_sources
    )
    seen = set()
    unique_sources: List[SourceRecord] = []
    for s in all_sources:
        if s.url and s.url in seen:
            errors.append(f"Duplicate source skipped: {s.url}")
            continue
        seen.add(s.url)
        unique_sources.append(s)

    # Step 2: Parse and chunk documents into passages
    retriever = Retriever(config=cfg)
    all_passages: List = []
    for source in unique_sources:
        content = source.metadata.get("summary", "") or source.metadata.get("description", "")
        if not content and source.url:
            content = f"Source: {source.title}. URL: {source.url}. Content not fetched in this prototype."
        try:
            passages = document_to_passages(source, content)
            all_passages.extend(passages)
        except Exception as exc:
            errors.append(f"Failed to parse {source.url}: {exc}")

    # Index passages for retrieval
    retriever.index_passages(all_passages)

    # Step 3: Build candidates from sources
    candidates: List[Candidate] = []
    for source in unique_sources:
        cand = Candidate(
            title=source.title,
            summary=source.metadata.get("summary", "") or source.metadata.get("description", "") or "",
            source=source,
            topics=topics,
            relevance_score=0.5,
        )
        candidates.append(cand)

    candidates = deduplicate_candidates(candidates)

    # Step 4: Verify evidence
    claims = {c.id: f"{c.title} is a significant development in {', '.join(topics)}" for c in candidates}
    verification_results = verify_claims_for_candidates(candidates, claims, config=cfg, retriever=retriever)

    # Step 5: ToT impact assessment
    assessment_result = impact_analyst(candidates, all_passages, verification_results, config=cfg)
    candidate_branches = assessment_result["candidate_branches"]
    rankings = assessment_result["rankings"]

    # Step 6: Final selections
    final_selections = []
    for entry in rankings:
        cand_id = entry["candidate_id"]
        candidate = next((c for c in candidates if c.id == cand_id), None)
        if candidate:
            final_selections.append({
                "candidate_id": cand_id,
                "title": candidate.title,
                "assessment": entry["best_assessment"],
                "score": entry["best_score"],
                "source_url": candidate.source.url,
            })

    return {
        "sources": unique_sources,
        "candidates": candidates,
        "passages": all_passages,
        "verification_results": verification_results,
        "candidate_branches": candidate_branches,
        "rankings": rankings,
        "final_selections": final_selections,
        "errors": errors,
        "termination_reason": "Workflow completed successfully",
        "config": cfg,
        "retriever": retriever,
    }