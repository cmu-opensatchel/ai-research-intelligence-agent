"""Flask web application for the AI Research Intelligence Agent.

Provides:
- Network graph visualization of AI domains across runs
- Domain detail pages with history and timeline
- Run listing and artifact browsing
- API endpoints for the frontend
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from flask import Flask, render_template, jsonify, request, redirect, url_for

from .config import RUN_ARTIFACTS_DIR, RUN_STATE_DB_PATH
from .domain_extractor import build_network_from_artifacts, extract_domain_history
from .storage import RunStore


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    store = RunStore()

    # Ensure directories exist
    RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/")
    def index():
        """Dashboard: show network graph and recent runs."""
        runs = store.list_runs()[:10]
        return render_template("dashboard.html", runs=runs)

    @app.route("/network")
    def network():
        """Network graph page."""
        return render_template("network.html")

    @app.route("/api/network")
    def api_network():
        """API: return merged network graph from all run artifacts."""
        artifacts: List = []
        for run in store.list_runs():
            artifact = store.load_artifact(run["run_id"])
            if artifact:
                artifacts.append(artifact)

        if not artifacts:
            return jsonify({"nodes": [], "edges": []})

        network = build_network_from_artifacts(artifacts)
        return jsonify(network)

    @app.route("/api/domain/<domain_id>")
    def api_domain(domain_id: str):
        """API: return history for a specific domain."""
        artifacts: List = []
        for run in store.list_runs():
            artifact = store.load_artifact(run["run_id"])
            if artifact:
                artifacts.append(artifact)

        history = extract_domain_history(artifacts, domain_id)
        return jsonify(history)

    @app.route("/api/runs")
    def api_runs():
        """API: list all runs."""
        runs = store.list_runs()
        return jsonify(runs)

    @app.route("/api/run/<run_id>")
    def api_run(run_id: str):
        """API: return a specific run artifact."""
        artifact = store.load_artifact(run_id)
        if not artifact:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(artifact.model_dump())

    @app.route("/domain/<domain_id>")
    def domain_detail(domain_id: str):
        """Domain detail page."""
        return render_template("domain.html", domain_id=domain_id)

    @app.route("/run/<run_id>")
    def run_detail(run_id: str):
        """Run detail page."""
        artifact = store.load_artifact(run_id)
        if not artifact:
            return "Run not found", 404
        return render_template("run.html", artifact=artifact, run_id=run_id)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)