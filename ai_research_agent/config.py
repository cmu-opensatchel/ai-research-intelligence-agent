"""Configuration for the AI Research Intelligence Agent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Default AI topics the system monitors (configurable, not hard-coded to any single domain)
DEFAULT_TOPICS: List[str] = [
    "agentic AI",
    "LLMs",
    "RAG",
    "multimodal AI",
    "GNNs",
]

# Default reporting window in days
DEFAULT_REPORTING_WINDOW_DAYS: int = 30

# Embedding model (lightweight, local)
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Vector store settings
VECTOR_STORE_PATH: Path = Path("data/vector_store")
COLLECTION_NAME: str = "ai_research_passages"

# Semantic retrieval settings
TOP_K_INITIAL: int = 5
TOP_K_FOLLOWUP: int = 5

# Evidence verification
EVIDENCE_SCORE_THRESHOLD: float = 0.5

# Tree of Thought settings
TOT_MAX_BRANCHES: int = 3
TOT_BEAM_WIDTH: int = 2
TOT_MAX_DEPTH: int = 2
TOT_MIN_SCORE: int = 50  # branches below this are pruned
TOT_MAX_FOLLOWUP_RETRIEVALS: int = 1

# Scoring rubric weights (must sum to 100)
SCORING_RUBRIC: dict = {
    "evidence_quality": 35,
    "relevance": 25,
    "novelty": 20,
    "practical_value": 20,
}

# Source types
SOURCE_TYPE_ARXIV = "arxiv"
SOURCE_TYPE_GITHUB = "github"
SOURCE_TYPE_LAB_ANNOUNCEMENT = "lab_announcement"
SOURCE_TYPE_TECHNICAL_REPORT = "technical_report"
SOURCE_TYPE_YOUTUBE = "youtube"
SOURCE_TYPE_X = "x"
SOURCE_TYPE_FACEBOOK = "facebook"
SOURCE_TYPE_INSTAGRAM = "instagram"
SOURCE_TYPE_LINKEDIN = "linkedin"
SOURCE_TYPE_HUGGINGFACE = "huggingface"
SOURCE_TYPE_CONFERENCE = "conference"

# Default lab RSS feed URLs (empty by default; users add their own)
DEFAULT_LAB_RSS_FEEDS: List[str] = []

# Database path for run state and history
RUN_STATE_DB_PATH: Path = Path("data/run_state.db")

# Run artifacts directory
RUN_ARTIFACTS_DIR: Path = Path("run_artifacts")

# HTTP request settings
REQUEST_TIMEOUT: int = 30
MAX_RETRIES: int = 2
USER_AGENT: str = "AI-Research-Intelligence-Agent/0.1 (research monitoring; contact: example@example.com)"

# Social media discovery (optional, opt-in)
# Social platforms are discovery signals only, not primary evidence.
# API credentials must be provided at runtime via environment variables;
# they are never stored in the repository.
ENABLE_SOCIAL_DISCOVERY: bool = False
SOCIAL_MAX_RESULTS_PER_PLATFORM: int = 5
YOUTUBE_API_KEY: str = ""  # set via YOUTUBE_API_KEY env var at runtime
X_BEARER_TOKEN: str = ""   # set via X_BEARER_TOKEN env var at runtime


@dataclass
class AgentConfig:
    """Runtime configuration for a workflow run."""

    topics: List[str] = field(default_factory=lambda: list(DEFAULT_TOPICS))
    reporting_window_days: int = DEFAULT_REPORTING_WINDOW_DAYS
    embedding_model: str = EMBEDDING_MODEL_NAME
    vector_store_path: Path = VECTOR_STORE_PATH
    collection_name: str = COLLECTION_NAME
    top_k_initial: int = TOP_K_INITIAL
    top_k_followup: int = TOP_K_FOLLOWUP
    evidence_score_threshold: float = EVIDENCE_SCORE_THRESHOLD
    tot_max_branches: int = TOT_MAX_BRANCHES
    tot_beam_width: int = TOT_BEAM_WIDTH
    tot_max_depth: int = TOT_MAX_DEPTH
    tot_min_score: int = TOT_MIN_SCORE
    scoring_rubric: dict = field(default_factory=lambda: dict(SCORING_RUBRIC))
    request_timeout: int = REQUEST_TIMEOUT
    max_retries: int = MAX_RETRIES
    user_agent: str = USER_AGENT
    enable_social_discovery: bool = ENABLE_SOCIAL_DISCOVERY
    social_max_results_per_platform: int = SOCIAL_MAX_RESULTS_PER_PLATFORM
    youtube_api_key: str = YOUTUBE_API_KEY
    x_bearer_token: str = X_BEARER_TOKEN
    rss_feed_urls: List[str] = field(default_factory=lambda: list(DEFAULT_LAB_RSS_FEEDS))