"""Data models for the AI Research Intelligence Agent.

All models distinguish retrieved facts (from public sources) from
model-generated interpretation/assessment.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    """A discovered public source with provenance metadata.

    publication_date: when the source was originally published.
    retrieval_time: when the agent collected it (distinct from publication date).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    url: str
    publication_date: Optional[str] = None
    retrieval_time: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_type: str  # e.g. "arxiv", "github", "lab_announcement"
    discovered_by: str  # agent role that found it
    metadata: dict = Field(default_factory=dict)


class Candidate(BaseModel):
    """A candidate development discovered from public sources."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    summary: str
    source: SourceRecord
    topics: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    status: str = "discovered"  # discovered, verified, ranked, selected, rejected


class Passage(BaseModel):
    """A retrieved text passage with full provenance.

    Exact text is preserved so users can inspect what grounded a response.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    source_url: str
    source_title: str
    section: Optional[str] = None
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class VerificationResult(BaseModel):
    """Result of checking whether a passage supports a claim.

    Distinguishes retrieved evidence from model interpretation.
    """

    claim: str
    supported: bool
    evidence_passages: List[Passage] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0
    insufficient_evidence: bool = False
    notes: str = ""


class ToTBranch(BaseModel):
    """A single branch in the bounded Tree of Thought search."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_id: str
    assessment: str  # e.g. "significant advance", "promising but limited", "insufficient evidence"
    supporting_evidence_refs: List[str] = Field(default_factory=list)  # passage IDs
    counterevidence: List[str] = Field(default_factory=list)
    uncertainty: str = ""
    score: int = 0  # 0-100
    depth: int = 0
    parent_id: Optional[str] = None
    pruned: bool = False
    prune_reason: Optional[str] = None


class RunArtifact(BaseModel):
    """Machine-readable record of a complete workflow run.

    Captures how retrieval and ToT affected the final output.
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    topics: List[str] = Field(default_factory=list)
    reporting_window_days: int = 0
    discovered_sources: List[SourceRecord] = Field(default_factory=list)
    candidates: List[Candidate] = Field(default_factory=list)
    retrieved_passages: List[Passage] = Field(default_factory=list)
    verification_results: List[VerificationResult] = Field(default_factory=list)
    tot_branches: List[ToTBranch] = Field(default_factory=list)
    final_selections: List[dict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    termination_reason: str = ""