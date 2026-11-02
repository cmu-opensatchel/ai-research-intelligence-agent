"""Deduplication and roundup report generation.

Combines verified candidates into a human-readable roundup and a
machine-readable run artifact. Each reported development shows:
- What is new
- Why it may matter
- Supporting evidence with source links
- Limitations or unresolved questions
- Recommended next step
- The assessment score or ranking rationale
"""

from typing import List, Optional

from .config import AgentConfig
from .models import Candidate, Passage, RunArtifact, SourceRecord, ToTBranch, VerificationResult
from .tot import rank_candidates


def deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """Remove duplicate candidates based on title similarity and source URL."""
    seen_titles = set()
    seen_urls = set()
    deduped: List[Candidate] = []

    for c in candidates:
        title_key = c.title.lower().strip()
        url_key = c.source.url.lower().strip() if c.source.url else ""
        if title_key in seen_titles:
            continue
        if url_key and url_key in seen_urls:
            continue
        seen_titles.add(title_key)
        if url_key:
            seen_urls.add(url_key)
        deduped.append(c)

    return deduped


def generate_roundup(
    candidates: List[Candidate],
    verification_results: List[VerificationResult],
    candidate_branches: dict,
    config: Optional[AgentConfig] = None,
) -> str:
    """Generate a human-readable roundup report."""
    cfg = config or AgentConfig()
    rankings = rank_candidates(candidate_branches, config=cfg)

    lines: List[str] = []
    lines.append("# AI Research Intelligence Roundup")
    lines.append(f"Topics: {', '.join(cfg.topics)}")
    lines.append(f"Reporting window: {cfg.reporting_window_days} days")
    lines.append("")

    if not rankings:
        lines.append("No significant developments found during this reporting period.")
        return "\n".join(lines)

    for rank, entry in enumerate(rankings, 1):
        cand_id = entry["candidate_id"]
        candidate = next((c for c in candidates if c.id == cand_id), None)
        if not candidate:
            continue

        vresult = next((v for v in verification_results if getattr(v, "claim", "").startswith(candidate.title)), None)

        lines.append(f"## {rank}. {candidate.title}")
        lines.append(f"**Assessment:** {entry['best_assessment']}")
        lines.append(f"**Score:** {entry['best_score']}/100")
        lines.append(f"**Source:** {candidate.source.title} ({candidate.source.url})")
        lines.append(f"**Source type:** {candidate.source.source_type}")
        lines.append(f"**Publication date:** {candidate.source.publication_date or 'Unknown'}")
        lines.append("")

        lines.append("**What is new:**")
        lines.append(candidate.summary)
        lines.append("")

        lines.append("**Why it may matter:**")
        if entry["best_assessment"] == "significant advance":
            lines.append("This development represents a notable advance in the topic area.")
        elif entry["best_assessment"] == "promising but limited":
            lines.append("This shows promise but has limitations that should be considered.")
        else:
            lines.append("Significance is unclear based on available evidence.")
        lines.append("")

        lines.append("**Supporting evidence:**")
        if vresult and vresult.evidence_passages:
            for p in vresult.evidence_passages[:3]:
                lines.append(f"- {p.text[:300]}... (Source: {p.source_title})")
        else:
            lines.append("- No supporting passages retrieved.")
        lines.append("")

        lines.append("**Limitations or unresolved questions:**")
        if vresult and vresult.insufficient_evidence:
            lines.append("- Evidence is insufficient to fully support the claim.")
        if vresult and vresult.contradictions:
            lines.append("- Contradictory language detected in source material.")
        if not vresult:
            lines.append("- No verification performed for this candidate.")
        lines.append("")

        lines.append("**Recommended next step:**")
        lines.append("- Review the primary source and retrieve additional context if needed.")
        lines.append("")

        lines.append("**Ranking rationale:**")
        lines.append(f"Score of {entry['best_score']}/100 based on evidence quality, relevance, novelty, and practical value.")
        lines.append("")

    return "\n".join(lines)


def build_run_artifact(
    run_id: str,
    topics: List[str],
    reporting_window_days: int,
    sources: List[SourceRecord],
    candidates: List[Candidate],
    passages: List[Passage],
    verification_results: List[VerificationResult],
    tot_branches: List[ToTBranch],
    final_selections: List[dict],
    errors: List[str],
    termination_reason: str,
) -> RunArtifact:
    """Build a machine-readable RunArtifact capturing the full workflow."""
    return RunArtifact(
        run_id=run_id,
        topics=topics,
        reporting_window_days=reporting_window_days,
        discovered_sources=sources,
        candidates=candidates,
        retrieved_passages=passages,
        verification_results=verification_results,
        tot_branches=tot_branches,
        final_selections=final_selections,
        errors=errors,
        termination_reason=termination_reason,
    )