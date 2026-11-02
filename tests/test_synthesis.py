"""Tests for deduplication and report generation."""

from ai_research_agent.models import Candidate, SourceRecord
from ai_research_agent.synthesis import deduplicate_candidates, generate_roundup


def make_candidate(cid, title, url):
    return Candidate(
        id=cid,
        title=title,
        summary=f"Summary for {title}",
        source=SourceRecord(title=title, url=url, source_type="arxiv", discovered_by="Research Scout"),
        topics=["agentic AI"],
    )


def test_deduplication_by_title():
    """Duplicate candidates with the same title should be removed."""
    candidates = [
        make_candidate("c1", "Same Paper", "https://example.com/1"),
        make_candidate("c2", "Same Paper", "https://example.com/2"),
        make_candidate("c3", "Different Paper", "https://example.com/3"),
    ]
    deduped = deduplicate_candidates(candidates)
    assert len(deduped) == 2


def test_deduplication_by_url():
    """Duplicate candidates with the same URL should be removed."""
    candidates = [
        make_candidate("c1", "Paper A", "https://example.com/same"),
        make_candidate("c2", "Paper B", "https://example.com/same"),
    ]
    deduped = deduplicate_candidates(candidates)
    assert len(deduped) == 1


def test_generate_roundup_no_candidates():
    """Roundup with no candidates should produce a sensible message."""
    report = generate_roundup([], {}, {}, None)
    assert "No significant developments" in report or "No significant" in report


def test_generate_roundup_with_candidates():
    """Roundup should include all required sections for each candidate."""
    from ai_research_agent.models import VerificationResult
    from ai_research_agent.tot import ToTBranch

    candidates = [make_candidate("c1", "AI Paper", "https://example.com/paper")]
    vresults = [
        VerificationResult(
            claim="AI Paper is significant",
            supported=True,
            confidence=0.8,
            evidence_passages=[],
        )
    ]
    branches = {
        "c1": [
            ToTBranch(candidate_id="c1", assessment="significant advance", score=85),
        ]
    }
    report = generate_roundup(candidates, vresults, branches, None)

    assert "AI Paper" in report
    assert "What is new" in report
    assert "Why it may matter" in report
    assert "Supporting evidence" in report
    assert "Limitations" in report
    assert "Recommended next step" in report
    assert "Ranking rationale" in report
    assert "85" in report  # score