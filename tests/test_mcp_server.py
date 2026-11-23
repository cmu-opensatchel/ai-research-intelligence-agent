"""Tests for the MCP server's SDK-independent logic (no mcp package needed)."""

import threading
import time
from types import SimpleNamespace

import pytest

from ai_research_agent.mcp_server import (
    ResearchService, ValidationError, parse_topics, validate_run_id, validate_window,
)
from ai_research_agent.models import Candidate, RunArtifact, SourceRecord
from ai_research_agent.storage import RunStore


def _cand(title="Paper A"):
    return Candidate(title=title, summary="s", source=SourceRecord(
        title="src", url="http://x/a", source_type="arxiv", discovered_by="t"))


def _result(cand):
    art = RunArtifact(topics=["rag"], reporting_window_days=7, candidates=[cand],
                      final_selections=[{"candidate_id": cand.id, "score": 80}],
                      termination_reason="done")
    return {"artifact": art, "roundup": "ROUNDUP", "candidates": [cand], "errors": [],
            "rankings": [{"candidate_id": cand.id, "best_assessment": "Significant", "best_score": 88}],
            "termination_reason": "done"}


@pytest.fixture
def svc(tmp_path):
    store = RunStore(db_path=tmp_path / "s.db", artifacts_dir=tmp_path / "a")
    return ResearchService(store=store, run_fn=lambda t, w, c: _result(_cand()))


def _wait(svc, job_id):
    for _ in range(100):
        s = svc.get_research_status(job_id)
        if s["status"] != "running":
            return s
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_topic_validation():
    assert parse_topics("agentic AI, RAG") == ["agentic AI", "RAG"]
    for bad in ["", " , ", "a;rm -rf", "x" * 101, ",".join("t%d" % i for i in range(11)), "<script>"]:
        with pytest.raises(ValidationError):
            parse_topics(bad)


def test_window_validation():
    assert validate_window(7) == 7
    for bad in [0, 366, -1, "7", True, 1.5]:
        with pytest.raises(ValidationError):
            validate_window(bad)


def test_run_id_rejects_path_traversal():
    for bad in ["../../etc/passwd", "run_1", "", "..%5cx", None]:
        with pytest.raises(ValidationError):
            validate_run_id(bad)


def test_start_and_complete(svc):
    job = svc.start_research("rag", 7)
    done = _wait(svc, job["job_id"])
    assert done["status"] == "completed"
    assert done["roundup"] == "ROUNDUP"
    assert done["rankings"][0] == {"title": "Paper A", "url": "http://x/a", "assessment": "Significant", "score": 88}


def test_failed_job_reported(tmp_path):
    def boom(t, w, c):
        raise RuntimeError("network down")
    s = ResearchService(store=RunStore(tmp_path / "s.db", tmp_path / "a"), run_fn=boom)
    done = _wait(s, s.start_research("rag")["job_id"])
    assert done["status"] == "failed" and "network down" in done["error"]


def test_only_one_job_at_a_time(tmp_path):
    gate = threading.Event()
    s = ResearchService(store=RunStore(tmp_path / "s.db", tmp_path / "a"),
                        run_fn=lambda t, w, c: (gate.wait(5), _result(_cand()))[1])
    first = s.start_research("rag")
    with pytest.raises(ValidationError):
        s.start_research("llms")
    gate.set()
    _wait(s, first["job_id"])
    s.start_research("llms")  # allowed again


def test_llm_flag_passed_to_config(tmp_path):
    seen = {}
    def fn(t, w, c):
        seen["llm"] = c.use_llm_verification
        return _result(_cand())
    s = ResearchService(store=RunStore(tmp_path / "s.db", tmp_path / "a"), run_fn=fn)
    _wait(s, s.start_research("rag", 7, True)["job_id"])
    assert seen["llm"] is True


def test_get_run_and_list_runs(svc):
    cand = _cand()
    art = _result(cand)["artifact"]
    svc.store.save_artifact(art)
    runs = svc.list_runs(5)
    assert runs[0]["run_id"] == art.run_id
    summary = svc.get_run(art.run_id)
    assert summary["counts"]["candidates"] == 1
    assert summary["final_selections"][0]["title"] == "Paper A"
    with pytest.raises(ValidationError):
        svc.get_run("00000000-0000-0000-0000-000000000000")


def test_unknown_job_and_claim_validation(svc):
    with pytest.raises(ValidationError):
        svc.get_research_status("00000000-0000-0000-0000-000000000000")
    for bad in ["", "   ", "x" * 1001, None]:
        with pytest.raises(ValidationError):
            svc.verify_claim(bad)
