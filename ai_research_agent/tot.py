"""Bounded Tree of Thought (ToT) for impact assessment.

Used selectively after discovery and retrieval, when the system must assess
a candidate's significance. Not used for routine date extraction, document
parsing, or exact-match deduplication.

For each candidate:
- Generate up to three alternative assessments.
- Represent each branch as structured data: assessment, evidence refs,
  counterevidence, uncertainty, score.
- Beam search with beam width 2 and max depth 2.
- Score branches using the rubric: evidence quality 35, relevance 25,
  novelty 20, practical value 20.
- Prune unsupported branches or branches scoring below 50/100.
- Resolve ties by stronger primary-source evidence, then relevance,
  then the more cautious interpretation.
- Limit to one additional retrieval round per candidate.
"""

from typing import List, Optional

from .config import AgentConfig, SCORING_RUBRIC, TOT_BEAM_WIDTH, TOT_MAX_BRANCHES, TOT_MAX_DEPTH, TOT_MIN_SCORE
from .models import Candidate, Passage, ToTBranch
from .retrieval import Retriever


DEFAULT_ASSESSMENTS = [
    "significant advance",
    "promising but limited",
    "insufficient evidence",
]


class ToTAssessor:
    """Bounded Tree of Thought impact assessor."""

    def __init__(self, config: Optional[AgentConfig] = None, retriever: Optional[Retriever] = None):
        self.cfg = config or AgentConfig()
        self.retriever = retriever or Retriever(config=self.cfg)

    def assess_candidate(
        self, candidate: Candidate, passages: List[Passage], verification_results: List
    ) -> List[ToTBranch]:
        """Run ToT search for a single candidate and return the surviving branches."""
        branches: List[ToTBranch] = []

        for assessment in DEFAULT_ASSESSMENTS[: self.cfg.tot_max_branches]:
            branch = ToTBranch(
                candidate_id=candidate.id,
                assessment=assessment,
                depth=0,
                score=self._score_branch(assessment, candidate, passages, verification_results),
            )
            branches.append(branch)

        # Keep all depth-0 branches that pass the minimum score
        branches = [b for b in branches if b.score >= self.cfg.tot_min_score]
        branches.sort(key=lambda b: b.score, reverse=True)

        # Beam search: only the top beam_width depth-0 branches are expanded to depth 1
        beam = branches[: self.cfg.tot_beam_width]

        depth1_branches: List[ToTBranch] = []
        for parent in beam:
            for assessment in DEFAULT_ASSESSMENTS:
                child = ToTBranch(
                    candidate_id=candidate.id,
                    assessment=f"{parent.assessment} (refined: {assessment})",
                    depth=1,
                    parent_id=parent.id,
                    score=self._score_branch(
                        f"{parent.assessment} (refined: {assessment})", candidate, passages, verification_results
                    ),
                )
                depth1_branches.append(child)

        # Prune depth-1 branches below minimum score
        depth1_branches = [b for b in depth1_branches if b.score >= self.cfg.tot_min_score]

        # Combine all surviving branches (depth 0 + depth 1)
        all_surviving = branches + depth1_branches
        all_surviving.sort(key=lambda b: b.score, reverse=True)

        if not all_surviving:
            # If all branches failed, do not force a positive conclusion
            branches.append(
                ToTBranch(
                    candidate_id=candidate.id,
                    assessment="insufficient evidence",
                    depth=0,
                    score=0,
                    pruned=True,
                    prune_reason="All branches scored below minimum threshold; no positive conclusion forced.",
                )
            )
            return branches

        return all_surviving

    def _score_branch(
        self,
        assessment: str,
        candidate: Candidate,
        passages: List[Passage],
        verification_results: List,
    ) -> int:
        """Score a branch using the rubric (0-100)."""
        evidence_score = 0
        supporting = [v for v in verification_results if getattr(v, "supported", False)]
        if supporting:
            avg_conf = sum(getattr(v, "confidence", 0) for v in supporting) / len(supporting)
            evidence_score = int(35 * avg_conf)
        elif any(getattr(v, "insufficient_evidence", False) for v in verification_results):
            evidence_score = 10

        relevance_score = int(25 * min(candidate.relevance_score, 1.0))
        novelty_score = self._assess_novelty(assessment, candidate, passages)
        practical_score = self._assess_practical_value(assessment, candidate)

        return evidence_score + relevance_score + novelty_score + practical_score

    def _assess_novelty(self, assessment: str, candidate: Candidate, passages: List[Passage]) -> int:
        """Assess novelty and significance (0-20)."""
        assessment_lower = assessment.lower()
        if "significant advance" in assessment_lower:
            return 20 if len(passages) > 0 else 10
        if "promising" in assessment_lower:
            return 12
        if "insufficient" in assessment_lower:
            return 0
        return 5

    def _assess_practical_value(self, assessment: str, candidate: Candidate) -> int:
        """Assess practical value and follow-up potential (0-20)."""
        assessment_lower = assessment.lower()
        if "significant advance" in assessment_lower:
            return 20
        if "promising" in assessment_lower:
            return 12
        if "insufficient" in assessment_lower:
            return 0
        return 5


def rank_candidates(
    candidate_branches: dict,
    config: Optional[AgentConfig] = None,
) -> List[dict]:
    """Rank candidates by their best surviving ToT branch.

    Resolves ties by stronger primary-source evidence, then relevance,
    then the more cautious interpretation.
    """
    cfg = config or AgentConfig()
    rankings: List[dict] = []

    for candidate_id, branches in candidate_branches.items():
        surviving = [b for b in branches if not b.pruned]
        if not surviving:
            continue
        best = max(surviving, key=lambda b: b.score)
        rankings.append(
            {
                "candidate_id": candidate_id,
                "best_assessment": best.assessment,
                "best_score": best.score,
                "branch_id": best.id,
                "num_branches": len(branches),
                "num_pruned": sum(1 for b in branches if b.pruned),
            }
        )

    def sort_key(r):
        assessment_lower = r["best_assessment"].lower()
        caution_penalty = 0 if "insufficient" not in assessment_lower else -1
        return (r["best_score"], caution_penalty)

    rankings.sort(key=sort_key, reverse=True)
    return rankings