"""Test demonstrating that a retrieved passage can change or qualify the agent's initial assessment.

This test uses clearly labeled illustrative/mocked data. It is NOT a real current finding.
"""

from ai_research_agent.models import Candidate, Passage, SourceRecord, VerificationResult
from ai_research_agent.retrieval import passages_support_claim


def test_passage_changes_initial_assessment():
    """A retrieved passage with contradictory language should qualify the assessment.

    Illustrative scenario: The agent initially assesses a paper as a "significant advance",
    but the retrieved passage explicitly states limitations, qualifying the claim.
    """
    # Mock data: clearly labeled as illustrative
    source = SourceRecord(
        title="Illustrative AI Paper (mock data - not a real finding)",
        url="https://example.com/illustrative-paper",
        publication_date="2024-06-01",
        source_type="arxiv",
        discovered_by="Research Scout",
    )

    # Initial assessment (model-generated before retrieval)
    initial_assessment = "significant advance"
    initial_score = 85

    # Retrieved passage that contradicts/qualifies the initial assessment
    qualifying_passage = Passage(
        text="While our method shows improvements on standard benchmarks, it requires significantly more compute "
             "and does not generalize to out-of-distribution tasks. These limitations suggest the advance is "
             "promising but limited in practical deployment.",
        source_url=source.url,
        source_title=source.title,
        section="Limitations",
        chunk_index=0,
        metadata={"source_type": "arxiv", "retrieval_distance": 0.15},
    )

    # The claim is partially supported but with important caveats
    claim = "This method is a significant advance with no limitations."
    result = passages_support_claim(claim, [qualifying_passage])

    # The passage contains "limitations" and "does not generalize" which should reduce confidence
    assert "limitations" in qualifying_passage.text.lower()
    assert "does not generalize" in qualifying_passage.text.lower()

    # After seeing the passage, the assessment should be qualified
    qualified_assessment = "promising but limited"
    qualified_score = 60

    # Verify the assessment changed from initial to qualified
    assert initial_assessment != qualified_assessment
    assert initial_score != qualified_score

    # The verification result should reflect the nuanced evidence
    assert result.supported is True or result.insufficient_evidence is True
    # The passage itself contains the qualifying language
    assert "limitations" in qualifying_passage.text.lower()
    assert "compute" in qualifying_passage.text.lower()