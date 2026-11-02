# AI Research Intelligence Agent

A multi-agent system for monitoring, verifying, and synthesizing emerging AI developments. This is a usable end-to-end prototype implementing a bounded workflow: public-source discovery → semantic retrieval → evidence verification → Tree of Thought (ToT) impact assessment → synthesis → run artifacts.

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Sample Run](#sample-run)
- [Inspecting Evidence and ToT Artifacts](#inspecting-evidence-and-tot-artifacts)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Safety](#safety)

## Architecture

The system is implemented as a Python package (`ai_research_agent/`) with separate, replaceable modules:

| Module | Responsibility |
|--------|---------------|
| `discovery.py` | Source discovery from arXiv, GitHub, and lab announcements |
| `parser.py` | Document parsing and chunking with metadata preservation |
| `retrieval.py` | Sentence-transformer embeddings + ChromaDB vector store, semantic search |
| `verification.py` | Evidence verification (source identifiability, claim support, date window, context) |
| `tot.py` | Bounded Tree of Thought impact assessment (beam width 2, depth 2) |
| `social_discovery.py` | Optional social media discovery (YouTube, X; Facebook/Instagram/LinkedIn placeholders) |
| `synthesis.py` | Deduplication and roundup report generation |
| `agents.py` | Agent roles: Research Scout, Industry Scout, Implementation Scout, Impact Analyst, Research Coordinator |
| `coordinator.py` | Workflow orchestration and run artifact creation |
| `storage.py` | SQLite run state + JSON artifact persistence |
| `cli.py` | Command-line interface (Typer + Rich) |

No CrewAI, LangChain, or MCP is used — the design uses the simplest tools that support the required behavior and is easy to extend.

### Agent Roles

- **Research Scout**: Discovers arXiv preprints and research papers matching configured topics.
- **Industry Scout**: Fetches lab announcements and technical reports.
- **Implementation Scout**: Discovers GitHub repositories and implementations.
- **Impact Analyst**: Runs ToT assessment and ranks candidates.
- **Research Coordinator**: Orchestrates the full end-to-end workflow.

## Installation

```bash
cd c:\Users\Mumu\Documents\CMU_AgenticAI\code
pip install -r requirements.txt
```

Requirements include: `requests`, `beautifulsoup4`, `feedparser`, `sentence-transformers`, `chromadb`, `rich`, `typer`, `pydantic`, `pytest`.

## Configuration

Configuration is in `ai_research_agent/config.py` (`AgentConfig` dataclass). Key settings:

```python
DEFAULT_TOPICS = ["agentic AI", "LLMs", "RAG", "multimodal AI", "GNNs"]
DEFAULT_REPORTING_WINDOW_DAYS = 30
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOT_BEAM_WIDTH = 2
TOT_MAX_DEPTH = 2
TOT_MIN_SCORE = 50
SCORING_RUBRIC = {"evidence_quality": 35, "relevance": 25, "novelty": 20, "practical_value": 20}
```

Topics and reporting window are configurable at runtime via the CLI. No credentials are stored in the repository.

## Usage

```bash
# Run with default topics and 30-day window
python -m ai_research_agent.cli run

# Run with custom topics and window
python -m ai_research_agent.cli run --topics "agentic AI,LLMs,RAG" --window 14

# Include lab announcement URLs
python -m ai_research_agent.cli run --topics "GNNs" --window 30 --urls "https://example.com/lab-announcement"

# View run history
python -m ai_research_agent.cli history
```

## Social Media Discovery (Optional)

Social media discovery is **disabled by default** and is opt-in via the `--social` flag. When enabled, the agent can discover signals from:

| Platform | API Required | Status |
|----------|-------------|--------|
| YouTube | `YOUTUBE_API_KEY` env var | Implemented (YouTube Data API v3) |
| X (Twitter) | `X_BEARER_TOKEN` env var | Implemented (API v2) |
| Facebook | OAuth app review | Not implemented (placeholder) |
| Instagram | OAuth app review | Not implemented (placeholder) |
| LinkedIn | OAuth app review | Not implemented (placeholder) |

**Important:** Social media results are **discovery signals only** — never treated as technical proof or primary evidence. Any claim surfaced from social media must be backed by a primary source (arXiv paper, lab announcement, technical report) before inclusion in the roundup.

```bash
# Enable social media discovery (requires API keys set as env vars)
export YOUTUBE_API_KEY=your_key
export X_BEARER_TOKEN=your_token
python -m ai_research_agent.cli run --topics "agentic AI" --window 7 --social
```

No credentials are stored in the repository. API keys/tokens must be supplied at runtime via environment variables.

## Sample Run

Running `python -m ai_research_agent.cli run --topics "agentic AI" --window 7` produces a roundup report showing each candidate's assessment, score, supporting evidence, limitations, and ranking rationale. A machine-readable run artifact is saved as `run_artifacts/run_<run_id>.json` and indexed in `data/run_state.db`.

## Inspecting Evidence and ToT Artifacts

Each run artifact JSON contains: `discovered_sources`, `candidates`, `retrieved_passages`, `verification_results`, `tot_branches`, `final_selections`, `errors`, and `termination_reason`.

To inspect a specific run:

```python
from ai_research_agent.storage import RunStore
store = RunStore()
runs = store.list_runs()
artifact = store.load_artifact(run_id)
```

Run artifacts are stored in `run_artifacts/` and the run index in `data/run_state.db`.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_tot.py -v
```

Tests cover:
- Chunking and metadata preservation
- Semantic retrieval returning relevant passages
- Claim-support checks and insufficient-evidence handling
- ToT branch limits, scoring, pruning, and tie-breaking
- Deduplication and report generation
- Handling unavailable sources or malformed responses
- A retrieved passage changing/qualifying the agent's initial assessment (illustrative mock data)

## Known Limitations

1. **Discovery is read-only**: arXiv API and GitHub Search API are used without authentication, which may be rate-limited.
2. **Content fetching is limited**: Full paper PDFs are not downloaded; arXiv abstracts and GitHub repository descriptions are used as content.
3. **Verification is heuristic**: Claim-support checking uses semantic similarity and keyword overlap, not LLM-based judgment. This is a prototype limitation.
4. **No web UI**: CLI only; web UI is optional and not implemented.
5. **Embedding model**: `all-MiniLM-L6-v2` is downloaded on first use (~80MB).
6. **Vector store**: Local ChromaDB instance; not distributed or production-scale.
7. **Social media**: Disabled by default. YouTube/X require API keys at runtime; Facebook, Instagram, LinkedIn are not implemented (require OAuth app review). Social results are discovery signals only, never primary evidence.

## Safety

- All retrieved data is treated as untrusted, never as instructions.
- No arbitrary code execution, unrestricted external actions, or confidential data.
- Bounded retries and narrow tool permissions are used.
- No credentials or secrets are stored in the repository.