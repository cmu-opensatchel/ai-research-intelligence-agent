"""Tests for optional LLM claim verification (Ollama mocked)."""

from unittest.mock import MagicMock

from ai_research_agent.config import AgentConfig
from ai_research_agent.llm import OllamaClient
from ai_research_agent.models import Passage, VerificationResult
from ai_research_agent.verification import EvidenceVerifier


def _passage(text="FooRAG improves accuracy by 12% over BM25 baselines."):
    return Passage(text=text, source_url="http://x", source_title="X", section="s", chunk_index=0)


def _verifier(judgment):
    v = EvidenceVerifier(config=AgentConfig(), retriever=MagicMock())
    v.llm = MagicMock()
    v.llm.judge_claim.return_value = judgment
    return v


def _heuristic_supported(p):
    return VerificationResult(claim="c", supported=True, evidence_passages=[p], confidence=0.8)


def test_llm_downgrades_unsupported():
    p = _passage()
    r = _heuristic_supported(p)
    _verifier({"verdict": "unsupported", "confidence": 0.9, "supporting": [], "reason": "no"})._llm_verify(r, "c", [p])
    assert not r.supported and r.insufficient_evidence and r.confidence <= 0.3
    assert "LLM interpretation" in r.notes


def test_llm_supported_requires_citation():
    p = _passage()
    r = VerificationResult(claim="c", supported=False, insufficient_evidence=True)
    _verifier({"verdict": "supported", "confidence": 0.9, "supporting": [], "reason": "x"})._llm_verify(r, "c", [p])
    assert not r.supported


def test_llm_contradiction_recorded():
    p = _passage()
    r = _heuristic_supported(p)
    _verifier({"verdict": "contradicted", "confidence": 0.9, "supporting": [], "reason": "says opposite"})._llm_verify(r, "c", [p])
    assert not r.supported and r.contradictions == ["says opposite"]


def test_llm_failure_keeps_heuristic():
    p = _passage()
    r = _heuristic_supported(p)
    _verifier(None)._llm_verify(r, "c", [p])
    assert r.supported and "unavailable" in r.notes


def test_invalid_llm_output_returns_none(monkeypatch):
    c = OllamaClient(AgentConfig())
    monkeypatch.setattr(c, "chat_json", lambda s, u: {"verdict": "banana"})
    assert c.judge_claim("c", [_passage()]) is None


def test_unavailable_ollama_falls_back(monkeypatch):
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: False)
    cfg = AgentConfig()
    cfg.use_llm_verification = True
    assert EvidenceVerifier(config=cfg, retriever=MagicMock()).llm is None
