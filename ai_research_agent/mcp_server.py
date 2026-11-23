"""MCP server exposing the AI Research Intelligence Agent to MCP clients.

Run over local stdio (never a network port):

    python -m ai_research_agent.mcp_server

Tools (all narrow; none fetch arbitrary URLs, read arbitrary paths, or run code):
- start_research: start a workflow run in the background, returns a job_id.
- get_research_status: poll a job; returns the roundup + top rankings when done.
- list_runs: recent saved runs.
- get_run: compact summary of one saved run (by run_id).
- verify_claim: check a claim against passages already in the local vector store.

All arguments come from an untrusted client and are validated here. The
workflow's retrieved web content is data, never instructions.
"""

import re
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig
from .storage import RunStore

MAX_TOPICS = 10
MAX_TOPIC_LEN = 100
MAX_CLAIM_LEN = 1000
MAX_WINDOW_DAYS = 365
MAX_LIST_RUNS = 50
MAX_ROUNDUP_CHARS = 8000
MAX_JOBS_KEPT = 20
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_TOPIC_RE = re.compile(r"^[\w\s.,+&/()\-]+$")


class ValidationError(ValueError):
    """Raised for invalid client-supplied arguments."""


def parse_topics(topics: str) -> List[str]:
    """Split and validate a comma-separated topic string."""
    parsed = [t.strip() for t in str(topics).split(",") if t.strip()]
    if not parsed:
        raise ValidationError("topics must contain at least one non-empty topic")
    if len(parsed) > MAX_TOPICS:
        raise ValidationError(f"at most {MAX_TOPICS} topics allowed")
    for t in parsed:
        if len(t) > MAX_TOPIC_LEN or not _TOPIC_RE.match(t):
            raise ValidationError(f"invalid topic: {t[:40]!r}")
    return parsed


def validate_window(window_days: int) -> int:
    if isinstance(window_days, bool) or not isinstance(window_days, int) or not 1 <= window_days <= MAX_WINDOW_DAYS:
        raise ValidationError(f"window_days must be an integer between 1 and {MAX_WINDOW_DAYS}")
    return window_days


def validate_run_id(run_id: str) -> str:
    """Run ids are UUIDs; anything else (e.g. path traversal) is rejected."""
    if not isinstance(run_id, str) or not _UUID_RE.match(run_id):
        raise ValidationError("run_id must be a UUID from list_runs")
    return run_id.lower()


def _default_run_fn(topics: List[str], window_days: int, config: AgentConfig) -> dict:
    from .coordinator import run_workflow

    return run_workflow(topics=topics, reporting_window_days=window_days, config=config)


class ResearchService:
    """SDK-independent logic behind the MCP tools (so it is easy to test)."""

    def __init__(
        self,
        store: Optional[RunStore] = None,
        run_fn: Callable[[List[str], int, AgentConfig], dict] = _default_run_fn,
    ):
        self.store = store or RunStore()
        self._run_fn = run_fn
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    # ---- background research jobs ----

    def start_research(self, topics: str, window_days: int = 7, use_llm_verification: bool = False) -> dict:
        topic_list = parse_topics(topics)
        window_days = validate_window(window_days)
        if not isinstance(use_llm_verification, bool):
            raise ValidationError("use_llm_verification must be a boolean")

        with self._lock:
            if any(j["status"] == "running" for j in self._jobs.values()):
                raise ValidationError("a research job is already running; poll it with get_research_status")
            job_id = str(uuid.uuid4())
            self._jobs[job_id] = {"status": "running", "topics": topic_list, "window_days": window_days}
            # Bound memory: drop the oldest finished jobs.
            while len(self._jobs) > MAX_JOBS_KEPT:
                oldest = next(k for k, j in self._jobs.items() if j["status"] != "running")
                del self._jobs[oldest]

        cfg = AgentConfig()
        cfg.use_llm_verification = use_llm_verification
        threading.Thread(target=self._run_job, args=(job_id, topic_list, window_days, cfg), daemon=True).start()
        return {"job_id": job_id, "status": "running", "topics": topic_list, "window_days": window_days}

    def _run_job(self, job_id: str, topics: List[str], window_days: int, cfg: AgentConfig) -> None:
        try:
            result = self._run_fn(topics, window_days, cfg)
            update = {
                "status": "completed",
                "run_id": result["artifact"].run_id,
                "roundup": str(result["roundup"])[:MAX_ROUNDUP_CHARS],
                "rankings": self._top_rankings(result),
                "errors": [str(e)[:300] for e in result.get("errors", [])][:10],
                "termination_reason": result.get("termination_reason", ""),
            }
        except Exception as e:  # report to the client instead of killing the thread silently
            update = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:500]}
        with self._lock:
            self._jobs[job_id].update(update)

    @staticmethod
    def _top_rankings(result: dict, limit: int = 10) -> List[dict]:
        lookup = {c.id: c for c in result.get("candidates", [])}
        out = []
        for entry in result.get("rankings", [])[:limit]:
            cand = lookup.get(entry["candidate_id"])
            out.append(
                {
                    "title": cand.title if cand else "",
                    "url": cand.source.url if cand and cand.source else "",
                    "assessment": entry.get("best_assessment"),
                    "score": entry.get("best_score"),
                }
            )
        return out

    def get_research_status(self, job_id: str) -> dict:
        validate_run_id(job_id)  # job ids are UUIDs too
        with self._lock:
            job = self._jobs.get(job_id.lower())
            if job is None:
                raise ValidationError("unknown job_id (jobs do not survive a server restart; see list_runs)")
            return {"job_id": job_id.lower(), **job}

    # ---- saved runs ----

    def list_runs(self, limit: int = 10) -> List[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_RUNS:
            raise ValidationError(f"limit must be an integer between 1 and {MAX_LIST_RUNS}")
        return self.store.list_runs()[:limit]

    def get_run(self, run_id: str) -> dict:
        run_id = validate_run_id(run_id)
        artifact = self.store.load_artifact(run_id)
        if artifact is None:
            raise ValidationError("no such run_id")
        titles = {c.id: c.title for c in artifact.candidates}
        return {
            "run_id": artifact.run_id,
            "timestamp": artifact.timestamp,
            "topics": artifact.topics,
            "window_days": artifact.reporting_window_days,
            "counts": {
                "sources": len(artifact.discovered_sources),
                "candidates": len(artifact.candidates),
                "passages": len(artifact.retrieved_passages),
                "verified_supported": sum(1 for v in artifact.verification_results if v.supported),
                "verification_results": len(artifact.verification_results),
            },
            "final_selections": [
                {**s, "title": titles.get(s.get("candidate_id", ""), "")} for s in artifact.final_selections[:10]
            ],
            "errors": [str(e)[:300] for e in artifact.errors][:10],
            "termination_reason": artifact.termination_reason,
        }

    # ---- claim verification ----

    def verify_claim(self, claim: str, use_llm_verification: bool = False) -> dict:
        """Check a claim against passages already indexed by earlier runs."""
        if not isinstance(claim, str) or not claim.strip() or len(claim) > MAX_CLAIM_LEN:
            raise ValidationError(f"claim must be a non-empty string up to {MAX_CLAIM_LEN} characters")
        if not isinstance(use_llm_verification, bool):
            raise ValidationError("use_llm_verification must be a boolean")

        from .models import Candidate, SourceRecord
        from .verification import EvidenceVerifier

        cfg = AgentConfig()
        cfg.use_llm_verification = use_llm_verification
        verifier = EvidenceVerifier(config=cfg)
        if verifier.retriever.count() == 0:
            raise ValidationError("the local evidence store is empty; run start_research first")
        stub = Candidate(
            title=claim[:80],
            summary=claim,
            source=SourceRecord(title="mcp client claim", url="", source_type="claim", discovered_by="mcp"),
        )
        result = verifier.verify_candidate(stub, claim.strip())
        return {
            "claim": result.claim,
            "supported": result.supported,
            "insufficient_evidence": result.insufficient_evidence,
            "confidence": round(result.confidence, 3),
            "contradictions": result.contradictions,
            "notes": result.notes.strip(),
            "evidence": [
                {"title": p.source_title, "url": p.source_url, "text": p.text[:500]}
                for p in result.evidence_passages[:5]
            ],
        }


def build_server(service: Optional[ResearchService] = None) -> Any:
    """Create the FastMCP server. Imports the SDK lazily so the rest of the package doesn't need it."""
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

    svc = service or ResearchService()
    mcp = FastMCP("ai-research-agent")

    @mcp.tool()
    def start_research(topics: str, window_days: int = 7, use_llm_verification: bool = False) -> dict:
        """Start a research run in the background (takes minutes). topics is comma-separated.
        Returns a job_id; poll get_research_status."""
        return svc.start_research(topics, window_days, use_llm_verification)

    @mcp.tool()
    def get_research_status(job_id: str) -> dict:
        """Status of a job from start_research; includes the roundup and top rankings when completed."""
        return svc.get_research_status(job_id)

    @mcp.tool()
    def list_runs(limit: int = 10) -> list:
        """List recent saved runs (newest first)."""
        return svc.list_runs(limit)

    @mcp.tool()
    def get_run(run_id: str) -> dict:
        """Compact summary of one saved run by run_id (from list_runs or a completed job)."""
        return svc.get_run(run_id)

    @mcp.tool()
    def verify_claim(claim: str, use_llm_verification: bool = False) -> dict:
        """Check a claim against passages already collected by earlier runs.
        Result is interpretation of retrieved evidence, not proof."""
        return svc.verify_claim(claim, use_llm_verification)

    return mcp


def main() -> None:
    build_server().run()  # stdio transport


if __name__ == "__main__":
    main()
