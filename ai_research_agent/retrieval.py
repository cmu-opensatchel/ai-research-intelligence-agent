"""Semantic retrieval pipeline using sentence-transformers embeddings and ChromaDB.

1. Create embeddings for passages and index them in a local vector store.
2. For a candidate claim, issue a focused semantic query and retrieve top-k chunks.
3. Preserve exact retrieved text, source URL, and metadata.
4. Allow one bounded follow-up retrieval of up to five more chunks if evidence is incomplete.
"""

import re
import uuid
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from .config import AgentConfig, COLLECTION_NAME, TOP_K_FOLLOWUP, TOP_K_INITIAL, VECTOR_STORE_PATH
from .models import Passage, VerificationResult


class Retriever:
    """Lightweight local semantic retrieval using embeddings + ChromaDB."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()
        self.embedding_model = SentenceTransformer(self.cfg.embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.cfg.vector_store_path))
        self.collection = self.client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_passages(self, passages: List[Passage]) -> int:
        """Embed and index a list of passages. Returns number indexed."""
        if not passages:
            return 0

        texts = [p.text for p in passages]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

        ids = [str(uuid.uuid4()) for _ in passages]
        metadatas = [
            {
                "source_url": p.source_url,
                "source_title": p.source_title,
                "section": p.section or "",
                "chunk_index": p.chunk_index,
                "source_type": p.metadata.get("source_type", ""),
                "publication_date": p.metadata.get("publication_date", "") or "",
                "retrieval_time": p.metadata.get("retrieval_time", ""),
                "discovered_by": p.metadata.get("discovered_by", ""),
            }
            for p in passages
        ]

        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return len(passages)

    def retrieve(self, query: str, top_k: int = TOP_K_INITIAL) -> List[Passage]:
        """Issue a focused semantic query and return top-k relevant passages."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        passages: List[Passage] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            passages.append(
                Passage(
                    text=doc,
                    source_url=meta.get("source_url", ""),
                    source_title=meta.get("source_title", ""),
                    section=meta.get("section") or None,
                    chunk_index=meta.get("chunk_index", 0),
                    metadata={
                        "source_type": meta.get("source_type", ""),
                        "publication_date": meta.get("publication_date") or None,
                        "retrieval_time": meta.get("retrieval_time", ""),
                        "discovered_by": meta.get("discovered_by", ""),
                        "retrieval_distance": float(dist),
                    },
                )
            )

        return passages

    def retrieve_for_claim(
        self, claim: str, candidate_title: str = ""
    ) -> tuple[List[Passage], List[Passage]]:
        """Retrieve passages for a claim, with one bounded follow-up round.

        Returns (initial_passages, followup_passages).
        The follow-up is only issued if initial results seem insufficient
        (e.g., very few results or low similarity).
        """
        initial = self.retrieve(claim, top_k=TOP_K_INITIAL)

        # Determine if a follow-up is warranted: too few results or all low similarity
        needs_followup = len(initial) < 3 or (
            initial and all((p.metadata.get("retrieval_distance", 1.0) or 1.0) > 0.6 for p in initial)
        )

        followup: List[Passage] = []
        if needs_followup and candidate_title:
            followup_query = f"{claim} {candidate_title}"
            followup = self.retrieve(followup_query, top_k=TOP_K_FOLLOWUP)

        return initial, followup

    def count(self) -> int:
        """Return the number of indexed passages."""
        return self.collection.count()

    def reset(self) -> None:
        """Clear the vector store collection (useful for tests)."""
        try:
            self.client.delete_collection(name=self.cfg.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def passages_support_claim(claim: str, passages: List[Passage]) -> VerificationResult:
    """Check whether retrieved passages support a claim.

    This is a heuristic check: a claim is considered supported if at least one
    retrieved passage has high semantic similarity to the claim and contains
    relevant keywords. The final judgment is recorded as model interpretation,
    not proof of scientific validity.
    """
    from .config import EVIDENCE_SCORE_THRESHOLD

    if not passages:
        return VerificationResult(
            claim=claim,
            supported=False,
            insufficient_evidence=True,
            confidence=0.0,
            notes="No passages retrieved for this claim.",
        )

    # Heuristic: check keyword overlap and similarity distance
    claim_lower = claim.lower()
    claim_words = set(w for w in re.split(r"\W+", claim_lower) if len(w) > 3)

    best_confidence = 0.0
    supported = False
    supporting: List[Passage] = []
    contradictions: List[str] = []

    for p in passages:
        text_lower = p.text.lower()
        passage_words = set(w for w in re.split(r"\W+", text_lower) if len(w) > 3)
        overlap = len(claim_words & passage_words) / max(len(claim_words), 1)

        dist = p.metadata.get("retrieval_distance", 1.0)
        if dist is None:
            dist = 1.0
        similarity = 1.0 - dist  # cosine distance -> similarity

        conf = 0.6 * similarity + 0.4 * overlap
        if conf > best_confidence:
            best_confidence = conf

        if conf >= EVIDENCE_SCORE_THRESHOLD:
            supported = True
            supporting.append(p)

        # Detect explicit contradiction language
        if any(w in text_lower for w in ["however", "but ", "contradict", "refute", "disprove", "not "]):
            contradictions.append(p.text[:200])

    if not supported and best_confidence < EVIDENCE_SCORE_THRESHOLD:
        insufficient = True
    else:
        insufficient = False

    return VerificationResult(
        claim=claim,
        supported=supported,
        evidence_passages=supporting,
        contradictions=contradictions,
        confidence=round(best_confidence, 3),
        insufficient_evidence=insufficient,
        notes=(
            "Passage(s) found with sufficient semantic overlap."
            if supported
            else "No passage sufficiently supports the claim."
        ),
    )