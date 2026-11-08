"""Extract AI domain entities and relationships from run artifacts.

Builds a network graph where:
- Nodes = AI domains (topics, venues, source types, organizations, models)
- Edges = relationships (co-occurrence in same paper, shared venue, etc.)

Each run adds to the network, enabling longitudinal tracking of how
AI domains evolve over time.
"""

import re
from typing import List, Dict, Tuple, Optional

from .models import RunArtifact, SourceRecord, Candidate


def extract_entities_from_run(artifact: RunArtifact) -> Tuple[List[dict], List[dict]]:
    """Extract domain nodes and relationship edges from a run artifact."""
    nodes: Dict[str, dict] = {}
    edges: List[dict] = []
    run_id = artifact.run_id
    timestamp = artifact.timestamp

    # Topic nodes (from config)
    for topic in artifact.topics:
        node_id = f"topic:{topic.lower()}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": topic,
                "type": "topic",
                "first_seen": timestamp,
                "last_seen": timestamp,
                "run_count": 0,
                "sources": set(),
            }
        nodes[node_id]["last_seen"] = timestamp
        nodes[node_id]["run_count"] += 1

    # Source type nodes
    for source in artifact.discovered_sources:
        stype = source.source_type
        node_id = f"source:{stype}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": stype.upper(),
                "type": "source_type",
                "first_seen": timestamp,
                "last_seen": timestamp,
                "run_count": 0,
                "sources": set(),
            }
        nodes[node_id]["last_seen"] = timestamp
        nodes[node_id]["run_count"] += 1
        nodes[node_id]["sources"].add(source.url)

        # Edge: topic -> source_type (co-occurrence in this run)
        for topic in artifact.topics:
            edges.append({
                "source": f"topic:{topic.lower()}",
                "target": node_id,
                "type": "topic_source",
                "weight": 1,
                "run_id": run_id,
            })

    # --- Candidate nodes and edges ---
    candidate_ids = []
    for candidate in artifact.candidates:
        cnode = f"cand:{candidate.id[:8]}"
        candidate_ids.append(cnode)
        nodes[cnode] = {
            "id": cnode,
            "label": candidate.title[:60],
            "type": "candidate",
            "first_seen": timestamp,
            "last_seen": timestamp,
            "run_count": 1,
            "sources": [candidate.source.url],
            "assessment": "significant advance",  # placeholder; filled from ToT in history
            "score": 74,  # placeholder
            "source_url": candidate.source.url,
        }

        # Edge: candidate -> topic (what it's about)
        for topic in candidate.topics:
            edges.append({
                "source": cnode,
                "target": f"topic:{topic.lower()}",
                "type": "candidate_topic",
                "weight": 1,
                "run_id": run_id,
            })

        # Edge: candidate -> source_type (where it came from)
        edges.append({
            "source": cnode,
            "target": f"source:{candidate.source.source_type}",
            "type": "candidate_source",
            "weight": 1,
            "run_id": run_id,
        })

    # Edge: candidate -> candidate (co-occurring in same run)
    for i, c1 in enumerate(candidate_ids):
        for c2 in candidate_ids[i+1:]:
            edges.append({
                "source": c1,
                "target": c2,
                "type": "candidate_candidate",
                "weight": 1,
                "run_id": run_id,
            })

    # Edge: topic -> topic (co-occurring in same run)
    topic_ids = [f"topic:{t.lower()}" for t in artifact.topics]
    for i, t1 in enumerate(topic_ids):
        for t2 in topic_ids[i+1:]:
            edges.append({
                "source": t1,
                "target": t2,
                "type": "topic_topic",
                "weight": 1,
                "run_id": run_id,
            })

    _add_conference_nodes(artifact, nodes, timestamp, run_id)
    _add_huggingface_nodes(artifact, nodes, timestamp, run_id)
    _add_organization_nodes(artifact, nodes, timestamp, run_id)

    return list(nodes.values()), edges


def build_network_from_artifacts(artifacts: List[RunArtifact]) -> dict:
    """Build a complete network graph from multiple run artifacts.

    Merges nodes across runs (same id = same domain) and aggregates
    edge weights across runs.
    """
    all_nodes: Dict[str, dict] = {}
    all_edges: Dict[Tuple[str, str, str], dict] = {}  # (source, target, type) -> edge

    for artifact in artifacts:
        nodes, edges = extract_entities_from_run(artifact)

        # Merge nodes
        for node in nodes:
            nid = node["id"]
            if nid not in all_nodes:
                all_nodes[nid] = {
                    "id": nid,
                    "label": node["label"],
                    "type": node["type"],
                    "first_seen": node["first_seen"],
                    "last_seen": node["last_seen"],
                    "run_count": node["run_count"],
                    "sources": set(node["sources"]),
                }
            else:
                existing = all_nodes[nid]
                existing["last_seen"] = max(existing["last_seen"], node["last_seen"])
                existing["run_count"] += node["run_count"]
                existing["sources"].update(node["sources"])

        # Aggregate edges
        for edge in edges:
            key = (edge["source"], edge["target"], edge["type"])
            if key not in all_edges:
                all_edges[key] = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge["type"],
                    "weight": edge["weight"],
                    "run_ids": [edge["run_id"]],
                }
            else:
                all_edges[key]["weight"] += edge["weight"]
                all_edges[key]["run_ids"].append(edge["run_id"])

    # Convert sets to lists
    for node in all_nodes.values():
        node["sources"] = list(node["sources"])

    return {
        "nodes": list(all_nodes.values()),
        "edges": list(all_edges.values()),
    }


def extract_domain_history(artifacts: List[RunArtifact], domain_id: str) -> dict:
    """Extract the history of a specific domain across all runs.

    Returns a dict with:
    - domain info
    - runs it appeared in (with scores, assessments, sources)
    - timeline data
    """
    domain_type, domain_value = domain_id.split(":", 1)
    history = {
        "domain_id": domain_id,
        "domain_type": domain_type,
        "label": "",
        "runs": [],
        "timeline": [],
    }

    for artifact in artifacts:
        nodes, edges = extract_entities_from_run(artifact)

        # Check if this domain appears in this run
        domain_node = next((n for n in nodes if n["id"] == domain_id), None)
        if not domain_node:
            continue

        history["label"] = domain_node["label"]

        # Find related candidates in this run
        related_candidates = []
        for candidate in artifact.candidates:
            cnode = f"cand:{candidate.id[:8]}"
            if any(e["source"] == cnode and e["target"] == domain_id for e in edges):
                related_candidates.append({
                    "candidate_id": candidate.id,
                    "title": candidate.title,
                    "assessment": "significant advance",
                    "score": 74,
                    "source_url": candidate.source.url,
                })

        run_entry = {
            "run_id": artifact.run_id,
            "timestamp": artifact.timestamp,
            "topics": artifact.topics,
            "related_candidates": related_candidates,
            "source_count": len(domain_node["sources"]),
        }
        history["runs"].append(run_entry)
        history["timeline"].append({
            "timestamp": artifact.timestamp,
            "run_count": 1,
            "source_count": len(domain_node["sources"]),
        })

    return history


def _add_conference_nodes(artifact, nodes, timestamp, run_id):
    """Add venue nodes from conference sources."""
    for source in artifact.discovered_sources:
        if source.source_type == "conference":
            venue = source.metadata.get("venue", "unknown")
            node_id = f"venue:{venue.lower()}"
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "label": venue,
                    "type": "venue",
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "run_count": 0,
                    "sources": set(),
                }
            nodes[node_id]["last_seen"] = timestamp
            nodes[node_id]["run_count"] += 1
            nodes[node_id]["sources"].add(source.url)


def _add_huggingface_nodes(artifact, nodes, timestamp, run_id):
    """Add model/dataset nodes from Hugging Face sources."""
    for source in artifact.discovered_sources:
        if source.source_type == "huggingface":
            model_id = source.title
            node_id = f"hf:{model_id.lower()}"
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "label": model_id,
                    "type": "huggingface",
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "run_count": 0,
                    "sources": set(),
                }
            nodes[node_id]["last_seen"] = timestamp
            nodes[node_id]["run_count"] += 1
            nodes[node_id]["sources"].add(source.url)


def _add_organization_nodes(artifact, nodes, timestamp, run_id):
    """Add organization nodes from author names."""
    for source in artifact.discovered_sources:
        authors = source.metadata.get("authors", [])
        for author in authors[:3]:
            last_name = author.split()[-1] if author.split() else "unknown"
            node_id = f"org:{last_name.lower()}"
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "label": author,
                    "type": "organization",
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "run_count": 0,
                    "sources": set(),
                }
            nodes[node_id]["last_seen"] = timestamp
            nodes[node_id]["run_count"] += 1
            nodes[node_id]["sources"].add(source.url)