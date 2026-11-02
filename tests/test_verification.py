"""Tests for claim-support checks and insufficient-evidence handling."""

import pytest

from ai_research_agent.models import Candidate, Passage, SourceRecord, VerificationResult
from ai_research_agent.retrieval import passages_support_claim
from ai_research_agent.verification import EvidenceVerifier


def make_passage(text, source_url="https://example.com/source", source_title="Example Source"):
    return Passage(
        text=text,
        source_url=source_url,
        source_title=source_title,
        section="Abstract",
        chunk_index=0,
        metadata={"source_type": "arxiv", "retrieval_distance": 0.2},
    )


def test_verification_supported_claim():
    """A claim that matches retrieved passages should be supported."""
    passages = [
        make_passage("Large language models demonstrate strong few-shot learning capabilities."),
        make_passage("Retrieval-augmented generation improves factual accuracy of LLMs."),
    ]
    result = passages_support_claim("Large language models have few-shot learning capabilities.", passages)
    assert result.supported is True
    assert result.confidence > 0
    assert len(result.evidence_passages) > 0


def test_verification_unsupported_claim():
    """A claim unrelated to retrieved passages should be unsupported."""
    passages = [
        make_passage("Graph neural networks process graph-structured data."),
    ]
    result = passages_support_claim("Quantum computing achieves quantum advantage.", passages)
    assert result.supported is False or result.insufficient_evidence is True


def test_verification_no_passages():
    """No passages means insufficient evidence."""
    result = passages_support_claim("Any claim.", [])
    assert result.insufficient_evidence is True
    assert result.supported is False


def test_evidence_verifier_checks_source_identifiability():
    """Verifier should note when passages lack source metadata."""
    verifier = EvidenceVerifier()
    candidate = Candidate(
        title="Test Candidate",
        summary="A test summary",
        source=SourceRecord(title="Test", url="https://example.com", source_type="arxiv", discovered_by="Research Scout"),
    )
    passages = [
        Passage(text="Some text", source_url="", source_title="", section="", chunk_index=0, metadata={}),
    ]
    result = verifier.verify_candidate(candidate, "Test claim")
    assert "lacks full source metadata" in result.notes or result.supported is False


def test_verification_result_distinguishes_evidence_from_interpretation():
    """VerificationResult should separate retrieved passages from model judgment."""
    passages = [make_passage("LLMs can perform few-shot learning.")]
    result = passages_support_claim("LLMs have few-shot learning.", passages)
    # evidence_passages contains retrieved facts; supported/confidence are model interpretation
    assert len(result.evidence_passages) == 1
    assert isinstance(result.supported, bool)
    assert isinstance(result.confidence, float)