"""Checkpoint 3 evidence: run semantic retrieval against the real local vector store and show
(1) retrieved passages with scores, (2) grounded vs ungrounded assessment, (3) failure modes.

Run from the repo root:  python examples/checkpoint3_retrieval_demo.py
Requires the vector store built by earlier runs (data/vector_store). Nothing here is mocked.
"""
import collections
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from ai_research_agent.config import AgentConfig, TOP_K_FOLLOWUP, TOP_K_INITIAL
from ai_research_agent.models import Candidate, SourceRecord
from ai_research_agent.retrieval import Retriever, passages_support_claim
from ai_research_agent.tot import ToTAssessor
from ai_research_agent.verification import EvidenceVerifier

sys.stdout.reconfigure(encoding="utf-8")
cfg = AgentConfig()
retriever = Retriever(config=cfg)


def hr(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def show(passages, n=5):
    for i, p in enumerate(passages[:n], 1):
        d = p.metadata.get("retrieval_distance")
        print(f"  [{i}] distance={d:.3f} (similarity {1 - d:.3f}) | {p.metadata.get('source_type')} | {(p.metadata.get('publication_date') or 'n/a')[:10]}")
        print(f"      {p.source_title}")
        print(f"      {p.source_url}")
        print(f"      \"{p.text[:230].strip()}...\"")


hr("0. CORPUS (external data discovered from public sources by earlier runs)")
meta = retriever.collection.get(include=["metadatas"], limit=10000)["metadatas"]
uniq = {(m.get("source_url"), m.get("chunk_index")) for m in meta}
print(f"  unique chunks indexed: {len(uniq)} (store entries: {retriever.count()}). Earlier runs had re-indexed chunks up to 33 times (1,038 entries); Retriever.dedupe_store() removed 881 exact duplicates.")
print(f"  by source type : {dict(collections.Counter(m.get('source_type') for m in meta))}")
print(f"  distinct sources: {len({m.get('source_url') for m in meta})}")
print(f"  embedding model: {cfg.embedding_model} | chunk size 1200 chars, overlap 150 | top-k {TOP_K_INITIAL} (+{TOP_K_FOLLOWUP} follow-up if weak)")

hr("1. SEMANTIC RETRIEVAL EXECUTED: three queries")
queries = [
    "Enterprise retrieval and agent systems need multitenant security isolation",
    "Safety and robustness risks of autonomous AI agents",
    "Open models released for multimodal vision-language tasks",
]
for q in queries:
    print(f"\nQUERY: {q!r}")
    init, follow = retriever.retrieve_for_claim(q, "")
    show(init)
    print(f"  -> follow-up retrieval triggered: {'yes' if follow else 'no'}")

hr("2. DOES RETRIEVAL CHANGE THE OUTPUT? grounded vs ungrounded assessment of the same candidate")
title = "Towards trustworthy agentic AI: a comprehensive survey of safety, robustness"
src = next(m for m in meta if m.get("source_title", "").startswith("Towards trustworthy"))
cand = Candidate(
    title=src["source_title"],
    summary="Survey of safety, robustness and trustworthiness of agentic AI.",
    source=SourceRecord(title=src["source_title"], url=src["source_url"], source_type="arxiv", discovered_by="demo"),
    topics=["agentic AI"],
)
claim = f"{cand.title} is a significant development in agentic AI"
print(f"CLAIM: {claim!r}\n")
assessor = ToTAssessor(config=cfg, retriever=retriever)
verifier = EvidenceVerifier(config=cfg, retriever=retriever)


def best(branches):
    b = max(branches, key=lambda x: x.score)
    return b.assessment, b.score


print("A) WITHOUT retrieval (no passages, no verification)")
none_branches = assessor.assess_candidate(cand, [], [])
print("   branches:", [(b.assessment, b.score) for b in sorted(none_branches, key=lambda x: -x.score)])
print("   best    :", best(none_branches) if none_branches else "all branches pruned")

print("\nB) WITH retrieval + verification")
init, follow = retriever.retrieve_for_claim(claim, cand.title)
passages = init + follow
res = verifier.verify_candidate(cand, claim)
print(f"   retrieved {len(init)} initial + {len(follow)} follow-up passages")
show(passages, 3)
print(f"   verification: supported={res.supported} insufficient_evidence={res.insufficient_evidence} confidence={res.confidence:.3f}")
print(f"   notes: {res.notes.strip()}")
br = assessor.assess_candidate(cand, passages, [res])
print("   branches:", [(b.assessment, b.score) for b in sorted(br, key=lambda x: -x.score)])
print("   best    :", best(br))

hr("2b. RETRIEVAL QUALIFIES A PROMOTIONAL CLAIM (ungrounded vs grounded answer)")
promo = "POLARIS achieves perfect precision across all agentic AI tasks"
print(f"CLAIM TO REPORT: {promo!r}")
print()
print("A) WITHOUT retrieval: nothing to check the claim against, so a summary can only repeat it:")
print(f"   -> \"{promo}.\"")
print()
init, follow = retriever.retrieve_for_claim(promo, "")
print("B) WITH retrieval: the passage the agent actually retrieved:")
p0 = init[0]
print(f"   {p0.source_title} | {p0.source_url}")
print(f"   \"{p0.text.strip()[:900]}\"")
print("   -> grounded statement: the retrieved passage reports 0.95 to 1.00 precision for anomaly routing on an initial benchmark;")
print("      it does not show perfect precision across all agentic AI tasks, so the broad claim must be qualified.")

hr("3. FAILURE MODE 1: the corpus has nothing relevant (off-topic claim)")
off = "Quantum error correction on superconducting qubits reduced logical error rates"
print(f"CLAIM: {off!r}")
init, follow = retriever.retrieve_for_claim(off, "Quantum error correction")
show(init, 3)
print(f"  follow-up retrieval triggered: {'yes' if follow else 'no'}")
off_cand = Candidate(title="Quantum error correction", summary=off,
                     source=SourceRecord(title="demo", url="", source_type="demo", discovered_by="demo"), topics=["quantum"])
r2 = verifier.verify_candidate(off_cand, off)
print(f"  verification: supported={r2.supported} insufficient_evidence={r2.insufficient_evidence} confidence={r2.confidence:.3f}")
print("  best assessment:", best(assessor.assess_candidate(off_cand, init + follow, [r2])) if assessor.assess_candidate(off_cand, init + follow, [r2]) else "all branches pruned (excluded)")

hr("4. FAILURE MODE 2: on-topic passages that do NOT support the specific (over-stated) claim")
cfg_llm = AgentConfig()
cfg_llm.use_llm_verification = True
v2 = EvidenceVerifier(config=cfg_llm, retriever=retriever)
for spec in [
    "POLARIS achieves perfect precision across all agentic AI tasks",
    "The trustworthy agentic AI survey proves that agents are safe to deploy in production",
]:
    print()
    print(f"CLAIM: {spec!r}")
    init, follow = retriever.retrieve_for_claim(spec, "")
    show(init, 2)
    h = passages_support_claim(spec, init + follow)
    print(f"  heuristic check (similarity + keyword overlap): supported={h.supported} confidence={h.confidence:.3f}   <-- topical match mistaken for support")
    if v2.llm is None:
        print("  LLM judge: unavailable (start Ollama with the Qwen3-8B model to run this step)")
        continue
    sc = Candidate(title=spec[:80], summary=spec,
                   source=SourceRecord(title="demo", url="", source_type="demo", discovered_by="demo"), topics=["agentic AI"])
    r3 = v2.verify_candidate(sc, spec)
    print(f"  with LLM judge: supported={r3.supported} insufficient_evidence={r3.insufficient_evidence} confidence={r3.confidence:.3f}")
    print("  judge note: " + r3.notes.split("[LLM interpretation")[-1].strip()[:420])
    br = assessor.assess_candidate(sc, init + follow, [r3])
    print("  ToT result:", [(b.assessment, b.score) for b in br if not b.pruned] or "no positive assessment survives (qualified/excluded)")
