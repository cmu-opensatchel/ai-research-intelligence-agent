"""Tests for ToT branch limits, scoring, pruning, and tie-breaking."""

import pytest

from ai_research_agent.config import AgentConfig
from ai_research_agent.models import Candidate, Passage, SourceRecord, ToTBranch, VerificationResult
from ai_research_agent.tot import ToTAssessor, rank_candidates


def make_candidate(cid, title, relevance_score=0.5):
    return Candidate(
        id=cid,
        title=title,
        summary="Test summary",
        source=SourceRecord(title="Test", url=f"https://example.com/{cid}", source_type="arxiv", discovered_by="Research Scout"),
        topics=["agentic AI"],
        relevance_score=relevance_score,
    )


def make_verification_result(supported=True, confidence=0.8, insufficient=False):
    return VerificationResult(
        claim="Test claim",
        supported=supported,
        confidence=confidence,
        insufficient_evidence=insufficient,
        evidence_passages=[],
    )


def test_tot_generates_three_assessments():
    """ToT should generate up to three alternative assessments per candidate."""
    assessor = ToTAssessor(config=AgentConfig(tot_max_branches=3, tot_min_score=0))
    candidate = make_candidate("c1", "Test AI Paper")
    passages = []
    vresults = [make_verification_result(supported=True, confidence=0.9)]
    branches = assessor.assess_candidate(candidate, passages, vresults)

    assessments = {b.assessment for b in branches}
    assert "significant advance" in assessments
    assert "promising but limited" in assessments
    assert "insufficient evidence" in assessments


def test_tot_prunes_below_min_score():
    """Branches scoring below the minimum threshold should be pruned."""
    config = AgentConfig(tot_min_score=80)  # high threshold
    assessor = ToTAssessor(config=config)
    candidate = make_candidate("c1", "Test AI Paper")
    passages = []
    vresults = [make_verification_result(supported=False, confidence=0.1)]
    branches = assessor.assess_candidate(candidate, passages, vresults)

    # All branches should be pruned or low-scoring
    assert all(b.score < 80 or b.pruned for b in branches)


def test_tot_beam_width_limits_depth1_expansion():
    """Beam width limits how many depth-0 branches are expanded to depth 1."""
    config = AgentConfig(tot_beam_width=2, tot_min_score=0)
    assessor = ToTAssessor(config=config)
    candidate = make_candidate("c1", "Test AI Paper")
    passages = []
    vresults = [make_verification_result(supported=True, confidence=0.9)]
    branches = assessor.assess_candidate(candidate, passages, vresults)

    # All 3 depth-0 assessments should be present
    depth0 = [b for b in branches if b.depth == 0]
    assert len(depth0) == 3

    # Only beam_width (2) depth-0 branches should have depth-1 children
    depth1 = [b for b in branches if b.depth == 1]
    parent_ids = {b.parent_id for b in depth1 if b.parent_id}
    assert len(parent_ids) <= config.tot_beam_width


def test_tot_no_positive_conclusion_when_all_fail():
    """If all branches fail, do not force a positive conclusion."""
    config = AgentConfig(tot_min_score=100)  # impossible threshold
    assessor = ToTAssessor(config=config)
    candidate = make_candidate("c1", "Test AI Paper")
    passages = []
    vresults = [make_verification_result(supported=False, confidence=0.0)]
    branches = assessor.assess_candidate(candidate, passages, vresults)

    # Should have an "insufficient evidence" branch, not a forced positive one
    assert any("insufficient" in b.assessment.lower() for b in branches)


def test_rank_candidates_sorting():
    """Rankings should sort by score descending."""
    candidate_branches = {
        "c1": [
            ToTBranch(candidate_id="c1", assessment="significant advance", score=85),
            ToTBranch(candidate_id="c1", assessment="promising but limited", score=60),
        ],
        "c2": [
            ToTBranch(candidate_id="c2", assessment="insufficient evidence", score=30),
            ToTBranch(candidate_id="c2", assessment="promising but limited", score=70),
        ],
    }
    rankings = rank_candidates(candidate_branches)
    assert len(rankings) == 2
    assert rankings[0]["best_score"] >= rankings[1]["best_score"]

def test_unsupported_retrieved_passages_do_not_yield_positive_conclusion():
    """Merely retrieving passages (e.g. off-topic ones) must not earn a positive assessment."""
    from ai_research_agent.models import Passage

    assessor = ToTAssessor(config=AgentConfig())
    candidate = make_candidate("c1", "Quantum error correction")
    passages = [Passage(text="unrelated text about image compression", source_url="http://x", source_title="X")]
    vresults = [make_verification_result(supported=False, confidence=0.1)]
    branches = assessor.assess_candidate(candidate, passages, vresults)
    assert not any("significant" in b.assessment.lower() for b in branches)


def test_high_relevance_cannot_rescue_unsupported_claim():
    """Even a highly relevant candidate gets no positive assessment when verification found no support."""
    from ai_research_agent.models import Passage

    assessor = ToTAssessor(config=AgentConfig())
    candidate = make_candidate("c2", "Off-topic claim", relevance_score=1.0)
    passages = [Passage(text="unrelated", source_url="http://x", source_title="X")]
    vresults = [make_verification_result(supported=False, confidence=0.1, insufficient=True)]
    branches = assessor.assess_candidate(candidate, passages, vresults)
    assert not any(("significant" in b.assessment.lower() or "promising" in b.assessment.lower()) and not b.pruned for b in branches)
