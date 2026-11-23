# AI Research Intelligence Agent

A multi-agent system for monitoring, verifying, and synthesizing emerging AI developments. This is a usable end-to-end prototype implementing a bounded workflow: public-source discovery → semantic retrieval → evidence verification → Tree of Thought (ToT) impact assessment → synthesis → run artifacts.

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Local LLM Verification (Qwen3-8B via Ollama)](#local-llm-verification-qwen3-8b-via-ollama)
- [Web UIs](#web-uis)
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
| `verification.py` | Evidence verification (source identifiability, claim support, date window, context) plus optional LLM judging |
| `llm.py` | Ollama client for local Qwen3-8B claim judging (never raises; falls back to heuristics) |
| `tot.py` | Bounded Tree of Thought impact assessment (beam width 2, depth 2) |
| `social_discovery.py` | Optional social media discovery (YouTube, X; Facebook/Instagram/LinkedIn placeholders) |
| `huggingface_discovery.py` | Hugging Face models/datasets discovery (public API, no key) |
| `conference_discovery.py` | Conference papers from OpenReview venues (NeurIPS, ICML, ICLR, etc.) |
| `rss_discovery.py` | Lab/blog RSS feed discovery (no key required) |
| `synthesis.py` | Deduplication and roundup report generation |
| `agents.py` | Agent roles: Research Scout, Industry Scout, Implementation Scout, Impact Analyst, Research Coordinator |
| `coordinator.py` | Workflow orchestration and run artifact creation |
| `storage.py` | SQLite run state + JSON artifact persistence |
| `cli.py` | Command-line interface (Typer + Rich) |
| `web.py` | Flask web UI with D3.js network graph |

At the repo root, `web_ui_streamlit.py` provides a Streamlit front end that runs the CLI.

No CrewAI, LangChain, or MCP is used — the design uses the simplest tools that support the required behavior and is easy to extend.

### Agent Roles

- **Research Scout**: Discovers arXiv preprints, conference papers, and research papers matching configured topics.
- **Industry Scout**: Fetches lab announcements and technical reports via RSS feeds.
- **Implementation Scout**: Discovers GitHub repositories, Hugging Face models/datasets, and implementations.
- **Impact Analyst**: Runs ToT assessment and ranks candidates.
- **Research Coordinator**: Orchestrates the full end-to-end workflow.

## Installation

```bash
cd c:\Users\Mumu\Documents\CMU_AgenticAI\code
pip install -r requirements.txt
```

Requirements include: `requests`, `beautifulsoup4`, `feedparser`, `sentence-transformers`, `chromadb`, `rich`, `typer`, `pydantic`, `pytest`, `flask`, `streamlit`.

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
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"
OLLAMA_TIMEOUT = 120
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

## Local LLM Verification (Qwen3-8B via Ollama)

Optional, off by default. After the heuristic check (embedding similarity + keyword overlap), a local Qwen3-8B model reads the retrieved passages and judges whether they support the claim.

Setup:

```bash
ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M   # model name must match OLLAMA_MODEL
python -m ai_research_agent.cli run --topics "agentic AI" --window 7 --use-llm-verification
```

Behavior:
- The model returns a verdict (`supported`, `partial`, `unsupported`, `contradicted`), a confidence, the numbered supporting passages, and a one-sentence reason.
- It can downgrade a claim or record a contradiction. It can mark a claim supported only if it cites at least one retrieved passage.
- Its verdict is appended to the result notes as `[LLM interpretation, <model>: ...]`, keeping model interpretation separate from retrieved evidence.
- If Ollama or the model is unavailable, or the output is invalid, a warning is logged and the heuristic result is used. A run never fails because of the LLM.
- Retrieved text is passed to the model as data only, and the model has no tools.

## Web UIs

```bash
# Streamlit UI (runs the CLI; has an "Enable LLM Verification" checkbox)
python -m streamlit run web_ui_streamlit.py     # http://localhost:8501

# Flask UI with D3.js network graph
python -m ai_research_agent.web
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

### Hugging Face Models & Datasets

Discovers models and datasets on Hugging Face Hub using the public API — no API key required:

```bash
python -m ai_research_agent.cli run --topics "agentic AI,LLMs" --window 30
```

The Implementation Scout queries HF for models and datasets matching each topic. Model cards and dataset pages are treated as primary sources but claims are still verified against associated papers.

### Conference Proceedings (NeurIPS, ICML, ICLR, etc.)

Discovers conference papers from OpenReview venues — no API key required:

```bash
python -m ai_research_agent.cli run --topics "agentic AI,GNNs" --window 30
```

Covers NeurIPS, ICML, ICLR, AISTATS, KDD, and other ML/AI venues indexed on OpenReview.

### Lab/Blog RSS Feeds

Configure RSS feed URLs for research-lab blogs. Add your feed URLs to `ai_research_agent/config.py` (`DEFAULT_LAB_RSS_FEEDS`) or pass them at runtime:

```python
# In config.py
DEFAULT_LAB_RSS_FEEDS = [
    "https://deepmind.google/blog/rss.xml",
    "https://ai.meta.com/blog/rss/",
    "https://openai.com/blog/rss/",
]
```

The Industry Scout fetches and parses each feed, extracting title, URL, publication date, and summary.

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
python -m pytest tests/ -v   # 55 tests

# Run a specific test file
python -m pytest tests/test_tot.py -v
```

Tests cover:
- Chunking and metadata preservation
- Semantic retrieval returning relevant passages
- Claim-support checks and insufficient-evidence handling
- ToT branch limits, scoring, pruning, and tie-breaking
- Deduplication and report generation
- LLM verification (Ollama mocked): downgrade, citation requirement, contradictions, fallback on failure or unavailability
- Handling unavailable sources or malformed responses
- A retrieved passage changing/qualifying the agent's initial assessment (illustrative mock data)

## Known Limitations

1. **Discovery is read-only**: arXiv API and GitHub Search API are used without authentication, which may be rate-limited.
2. **Content fetching is limited**: Full paper PDFs are not downloaded; arXiv abstracts and GitHub repository descriptions are used as content.
3. **Verification is mostly heuristic**: By default claim support uses semantic similarity and keyword overlap. The optional local LLM judge (`--use-llm-verification`) improves this but is an 8B quantized model reading at most 6 passages per claim; its verdicts are interpretation, not proof, and it adds latency.
4. **Web UIs are thin**: The Streamlit UI shells out to the CLI and parses its text output; the Flask UI is a visualization aid.
5. **Embedding model**: `all-MiniLM-L6-v2` is downloaded on first use (~80MB).
6. **Vector store**: Local ChromaDB instance; not distributed or production-scale.
7. **Social media**: Disabled by default. YouTube/X require API keys at runtime; Facebook, Instagram, LinkedIn are not implemented (require OAuth app review). Social results are discovery signals only, never primary evidence.

## Safety

- All retrieved data is treated as untrusted, never as instructions.
- No arbitrary code execution, unrestricted external actions, or confidential data.
- Bounded retries and narrow tool permissions are used.
- No credentials or secrets are stored in the repository.