"""Tests for semantic retrieval returning relevant passages."""

import os
import tempfile
import pytest

from ai_research_agent.config import AgentConfig, VECTOR_STORE_PATH
from ai_research_agent.models import Passage, SourceRecord
from ai_research_agent.retrieval import Retriever, passages_support_claim


@pytest.fixture
def temp_config():
    """Create a config with a temporary vector store path."""
    tmpdir = tempfile.mkdtemp()
    cfg = AgentConfig(vector_store_path=__import__("pathlib").Path(tmpdir) / "vector_store")
    yield cfg
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def retriever(temp_config):
    """Create a Retriever with a temp vector store."""
    r = Retriever(config=temp_config)
    yield r
    r.reset()


@pytest.fixture
def sample_passages():
    """Create sample passages about AI topics."""
    source = SourceRecord(
        title="AI Paper",
        url="https://arxiv.org/abs/2401.12345",
        publication_date="2024-01-15",
        source_type="arxiv",
        discovered_by="Research Scout",
    )
    return [
        Passage(
            text="Large language models have demonstrated remarkable capabilities in few-shot learning and reasoning tasks.",
            source_url=source.url,
            source_title=source.title,
            section="Abstract",
            chunk_index=0,
            metadata={"source_type": "arxiv"},
        ),
        Passage(
            text="Retrieval-augmented generation combines LLMs with external knowledge bases to improve factual accuracy.",
            source_url=source.url,
            source_title=source.title,
            section="Introduction",
            chunk_index=1,
            metadata={"source_type": "arxiv"},
        ),
        Passage(
            text="Graph neural networks are effective for learning on graph-structured data such as molecular structures.",
            source_url=source.url,
            source_title=source.title,
            section="Related Work",
            chunk_index=2,
            metadata={"source_type": "arxiv"},
        ),
    ]


def test_retriever_index_and_retrieve(retriever, sample_passages):
    """Test that indexed passages can be retrieved via semantic search."""
    n_indexed = retriever.index_passages(sample_passages)
    assert n_indexed == len(sample_passages)
    assert retriever.count() == len(sample_passages)

    results = retriever.retrieve("large language models few-shot learning", top_k=3)
    assert len(results) > 0
    # The first result should be about LLMs
    assert "language model" in results[0].text.lower() or "llm" in results[0].text.lower()


def test_retriever_preserves_metadata(retriever, sample_passages):
    """Test that retrieved passages preserve source URL and metadata."""
    retriever.index_passages(sample_passages)
    results = retriever.retrieve("language models", top_k=1)
    assert len(results) == 1
    assert results[0].source_url == "https://arxiv.org/abs/2401.12345"
    assert results[0].source_title == "AI Paper"
    assert results[0].metadata["source_type"] == "arxiv"


def test_passages_support_claim_with_evidence(retriever, sample_passages):
    """Test that a supported claim returns supported=True."""
    retriever.index_passages(sample_passages)
    passages = retriever.retrieve("large language models few-shot learning", top_k=3)
    result = passages_support_claim("Large language models have few-shot learning capabilities.", passages)
    assert result.supported is True
    assert result.confidence > 0


def test_passages_support_claim_insufficient_evidence(retriever, sample_passages):
    """Test that an unsupported claim returns insufficient_evidence=True."""
    retriever.index_passages(sample_passages)
    passages = retriever.retrieve("quantum computing algorithms", top_k=3)
    result = passages_support_claim("Quantum computing can factor large integers efficiently.", passages)
    # Should not be well supported since passages are about AI, not quantum
    assert result.supported is False or result.insufficient_evidence is True


def test_passages_support_claim_no_passages():
    """Test that no passages returns insufficient evidence."""
    result = passages_support_claim("Some claim.", [])
    assert result.supported is False
    assert result.insufficient_evidence is True
    assert result.confidence == 0.0

def test_index_is_idempotent(retriever, sample_passages):
    """Re-indexing the same chunks must not create duplicates."""
    retriever.index_passages(sample_passages)
    n = retriever.count()
    retriever.index_passages(sample_passages)
    assert retriever.count() == n


def test_retrieve_dedupes_legacy_duplicates(retriever, sample_passages):
    """Top-k must not be filled with copies of one chunk (older runs indexed chunks repeatedly)."""
    retriever.index_passages(sample_passages)
    got = retriever.collection.get(include=["documents", "metadatas", "embeddings"], limit=1)
    retriever.collection.add(
        ids=["legacy-dup-1", "legacy-dup-2"],
        documents=got["documents"] * 2,
        metadatas=got["metadatas"] * 2,
        embeddings=[list(got["embeddings"][0])] * 2,
    )
    results = retriever.retrieve(got["documents"][0], top_k=5)
    texts = [p.text for p in results]
    assert len(texts) == len(set(texts))


def test_retrieve_returns_k_distinct_even_with_many_copies(retriever, sample_passages):
    """With many copies of a chunk stored, top-k still returns up to k distinct passages."""
    retriever.index_passages(sample_passages)
    got = retriever.collection.get(include=["documents", "metadatas", "embeddings"], limit=1)
    n = 6
    retriever.collection.add(
        ids=[f"copy-{i}" for i in range(n)],
        documents=got["documents"] * n,
        metadatas=got["metadatas"] * n,
        embeddings=[list(got["embeddings"][0])] * n,
    )
    results = retriever.retrieve(got["documents"][0], top_k=3)
    assert len({p.text for p in results}) == min(3, len(sample_passages))


def test_dedupe_store_removes_duplicates(retriever, sample_passages):
    retriever.index_passages(sample_passages)
    base = retriever.count()
    got = retriever.collection.get(include=["documents", "metadatas", "embeddings"], limit=1)
    retriever.collection.add(
        ids=["d1", "d2"], documents=got["documents"] * 2, metadatas=got["metadatas"] * 2,
        embeddings=[list(got["embeddings"][0])] * 2,
    )
    assert retriever.dedupe_store() == 2
    assert retriever.count() == base
