"""Workflow coordinator that ties all modules together and produces run artifacts."""

from typing import Optional

from .agents import research_coordinator
from .config import AgentConfig
from .models import RunArtifact
from .synthesis import build_run_artifact, generate_roundup
from .storage import RunStore


def run_workflow(
    topics: list,
    reporting_window_days: int = 30,
    announcement_urls: Optional[list] = None,
    config: Optional[AgentConfig] = None,
    save_artifact: bool = True,
) -> dict:
    """Run the complete AI research intelligence workflow.

    Steps:
    1. Configure topics and reporting period
    2. Discover public sources
    3. Retrieve relevant passages
    4. Verify evidence
    5. Assess and rank candidates (ToT)
    6. Synthesize roundup
    7. Save run artifacts

    Returns a dict with the roundup report, run artifact, and all intermediate results.
    """
    cfg = config or AgentConfig()
    cfg.topics = topics
    cfg.reporting_window_days = reporting_window_days

    # Run the agent coordinator
    result = research_coordinator(
        topics=topics,
        reporting_window_days=reporting_window_days,
        announcement_urls=announcement_urls,
        config=cfg,
    )

    # Generate roundup report
    roundup = generate_roundup(
        candidates=result["candidates"],
        verification_results=result["verification_results"],
        candidate_branches=result["candidate_branches"],
        config=cfg,
    )

    from uuid import uuid4
    run_id = str(uuid4())

    # Build run artifact
    artifact = build_run_artifact(
        run_id=run_id,
        topics=topics,
        reporting_window_days=reporting_window_days,
        sources=result["sources"],
        candidates=result["candidates"],
        passages=result["passages"],
        verification_results=result["verification_results"],
        tot_branches=[b for branches in result["candidate_branches"].values() for b in branches],
        final_selections=result["final_selections"],
        errors=result["errors"],
        termination_reason=result["termination_reason"],
    )

    # Save artifact
    artifact_path = None
    if save_artifact:
        store = RunStore()
        artifact_path = store.save_artifact(artifact)

    return {
        "roundup": roundup,
        "artifact": artifact,
        "artifact_path": artifact_path,
        "sources": result["sources"],
        "candidates": result["candidates"],
        "passages": result["passages"],
        "verification_results": result["verification_results"],
        "candidate_branches": result["candidate_branches"],
        "rankings": result["rankings"],
        "final_selections": result["final_selections"],
        "errors": result["errors"],
        "termination_reason": result["termination_reason"],
    }