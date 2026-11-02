"""Run state and history storage using SQLite and JSON artifacts."""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .config import RUN_ARTIFACTS_DIR, RUN_STATE_DB_PATH
from .models import RunArtifact


class RunStore:
    """Persists run artifacts and history in SQLite and JSON files."""

    def __init__(self, db_path: Path = RUN_STATE_DB_PATH, artifacts_dir: Path = RUN_ARTIFACTS_DIR):
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    reporting_window_days INTEGER NOT NULL,
                    artifact_path TEXT NOT NULL,
                    termination_reason TEXT
                )
                """
            )
            conn.commit()

    def save_artifact(self, artifact: RunArtifact) -> Path:
        """Save a run artifact as JSON and index it in SQLite."""
        artifact_path = self.artifacts_dir / f"run_{artifact.run_id}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact.model_dump(), f, indent=2, default=str)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, timestamp, topics, reporting_window_days, artifact_path, termination_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.run_id,
                    artifact.timestamp,
                    json.dumps(artifact.topics),
                    artifact.reporting_window_days,
                    str(artifact_path),
                    artifact.termination_reason,
                ),
            )
            conn.commit()
        return artifact_path

    def load_artifact(self, run_id: str) -> Optional[RunArtifact]:
        """Load a run artifact from its JSON file."""
        artifact_path = self.artifacts_dir / f"run_{run_id}.json"
        if not artifact_path.exists():
            return None
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RunArtifact(**data)

    def list_runs(self) -> List[dict]:
        """List all recorded runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, timestamp, topics, reporting_window_days, termination_reason FROM runs ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]