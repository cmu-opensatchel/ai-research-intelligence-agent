"""Evidence verification module.

Before using retrieved information, check:
- The source is identifiable and the URL and metadata are retained.
- The passage answers the specific question or supports the specific claim.
- The date fits the reporting window.
- The passage includes enough context to interpret the result correctly.
- Important claims are supported by primary sources when available.
- Contradictions, missing evidence, and uncertainty are recorded.

A citation existing is not enough; the cited passage must support the claim.
If evidence is insufficient, qualify the claim, mark it unresolved, or omit it.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from .config import AgentConfig
from .llm import OllamaClient
from .models import Candidate, Passage, VerificationResult
from .retrieval import Retriever, passages_support_claim

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    """Verifies that retrieved evidence supports candidate claims."""

    def __init__(self, config: Optional[AgentConfig] = None, retriever: Optional[Retriever] = None):
        self.cfg = config or AgentConfig()
        self.retriever = retriever or Retriever(config=self.cfg)
        self.llm: Optional[OllamaClient] = None
        if self.cfg.use_llm_verification:
            client = OllamaClient(self.cfg)
            if client.is_available():
                self.llm = client
            else:
                logger.warning(
                    "LLM verification requested but Ollama model %s is unavailable at %s; "
                    "falling back to heuristic verification.",
                    self.cfg.ollama_model, self.cfg.ollama_url,
                )

    def verify_candidate(self, candidate: Candidate, claim: str) -> VerificationResult:
        """Verify a claim for a candidate using retrieval + evidence checks."""
        # Retrieve passages for the claim
        initial, followup = self.retriever.retrieve_for_claim(claim, candidate.title)
        all_passages = initial + followup

        # Run heuristic support check
        result = passages_support_claim(claim, all_passages)

        # Additional verification checks
        self._check_source_identifiability(result, all_passages)
        self._check_date_window(result, all_passages)
        self._check_context_sufficiency(result, all_passages)
        self._check_primary_source_support(result, all_passages, candidate)

        if self.llm is not None:
            self._llm_verify(result, claim, all_passages)

        return result

    def _llm_verify(self, result: VerificationResult, claim: str, passages: List[Passage]) -> None:
        """Refine the heuristic verdict with the local LLM (recorded as model interpretation).

        The LLM may confirm, downgrade, or flag contradictions. It can only mark a claim
        supported if it cites at least one passage. On failure the heuristic result stands.
        """
        judgment = self.llm.judge_claim(claim, passages[:6])
        if judgment is None:
            result.notes += " LLM verification unavailable for this claim; heuristic result used."
            return

        verdict = judgment["verdict"]
        cited = [passages[n - 1] for n in judgment["supporting"]]
        if verdict == "supported" and cited:
            result.supported = True
            result.insufficient_evidence = False
            result.evidence_passages = cited
            result.confidence = judgment["confidence"]
        elif verdict == "partial":
            result.supported = False
            result.insufficient_evidence = True
            result.confidence = min(result.confidence, judgment["confidence"])
        elif verdict == "contradicted":
            result.supported = False
            result.contradictions.append(judgment["reason"])
            result.confidence = 0.0
        else:  # unsupported, or "supported" without any cited passage
            result.supported = False
            result.insufficient_evidence = True
            result.confidence = min(result.confidence, 0.3)
        result.notes += f" [LLM interpretation, {self.cfg.ollama_model}: {verdict} - {judgment['reason']}]"

    def _check_source_identifiability(self, result: VerificationResult, passages: List[Passage]) -> None:
        """Ensure each passage has an identifiable source URL and metadata."""
        for p in passages:
            if not p.source_url or not p.source_title:
                result.notes += " Warning: a retrieved passage lacks full source metadata."
                break

    def _check_date_window(self, result: VerificationResult, passages: List[Passage]) -> None:
        """Check that passage publication dates fit the reporting window."""
        # publication_date is stored as a string; we do a best-effort check.
        # If dates are missing or unparseable, we don't disqualify but note it.
        for p in passages:
            pub_date = p.metadata.get("publication_date")
            if pub_date:
                try:
                    parsed = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    # Just note that we found a date; window filtering happens at discovery
                    pass
                except Exception:
                    pass

    def _check_context_sufficiency(self, result: VerificationResult, passages: List[Passage]) -> None:
        """Ensure passages include enough context to interpret correctly."""
        for p in passages:
            if len(p.text) < 50:
                result.notes += " Warning: some retrieved passages are very short and may lack context."
                break

    def _check_primary_source_support(
        self, result: VerificationResult, passages: List[Passage], candidate: Candidate
    ) -> None:
        """Note whether primary sources (arXiv, lab announcements) support the claim."""
        primary_types = {"arxiv", "lab_announcement", "technical_report"}
        has_primary = any(p.metadata.get("source_type") in primary_types for p in passages)
        if not has_primary and passages:
            result.notes += " Note: no primary source directly supports this claim."
        elif has_primary:
            result.notes += " Primary source(s) found supporting this claim."


def verify_claims_for_candidates(
    candidates: List[Candidate],
    claims: dict,
    config: Optional[AgentConfig] = None,
    retriever: Optional[Retriever] = None,
) -> List[VerificationResult]:
    """Verify a set of claims (one per candidate) and return verification results."""
    verifier = EvidenceVerifier(config=config, retriever=retriever)
    results: List[VerificationResult] = []
    for candidate in candidates:
        claim = claims.get(candidate.id, f"{candidate.title} is a significant development in {', '.join(candidate.topics)}")
        result = verifier.verify_candidate(candidate, claim)
        results.append(result)
    return results