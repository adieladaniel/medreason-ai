"""
The retrieval entry point the evidence planner and explanation writer call.

Pipeline (each choice measured in app/rag/eval_pipeline.py on 54 queries split into dev and
test; experiments/rag_retrieval/PIPELINE.md has the numbers):

  1. hybrid search: dense embeddings (NeuML/pubmedbert-base-embeddings, in Qdrant) and BM25
     keyword search, merged by reciprocal rank fusion
  2. rerank the top 5 with a biomedical cross-encoder (ncbi/MedCPT-Cross-Encoder)
  3. drop passages the reranker scores below MIN_RERANK_SCORE

Step 3 is how the system says "no supporting evidence": an empty list means nothing in the
corpus bears on the query, and the explanation must say so instead of citing weak matches.
The threshold is on the reranker's raw logit, not the dense cosine used in stage 1 and not a
sigmoid probability (those saturate near 1.0, which makes any threshold hair-thin).

Query expansion (index/hybrid.expand_query) is deliberately not applied: it did not help on
dev or test, with or without hybrid search or reranking.

Known limits, so nobody over-trusts this: the reranker can confidently rank a near-miss first
(for "platelet-to-lymphocyte ratio in pneumonia" it put an NLR passage ahead of the paper on
exactly that), and a high score means "this passage is on the query's topic", not "this
passage supports the claim". The writer must check what a passage actually states.
"""
from __future__ import annotations

import torch  # noqa: F401

from app.rag.hybrid import Retriever

RERANKER = "ncbi/MedCPT-Cross-Encoder"
# Midpoint between the highest dev off-corpus score (2.86) and the lowest dev answerable score
# (5.34), rounded. Chosen on dev only; on test it refused 8 of 8 off-corpus queries and kept
# 26 of 28 answerable ones at the unrounded dev threshold. Provisional: 16 off-corpus queries.
MIN_RERANK_SCORE = 4.0
# Reranking the top 5 matched reranking the top 10 or 20 on dev and test (p@1, p@3, abstention
# AUC), and the cross-encoder is the whole latency (about 0.7 s per passage on this CPU).
CANDIDATES = 5

_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Up to k passages, best first, each with its citation and its reranker score."""
    results = retriever().search(query, k=CANDIDATES, mode="hybrid", rerank=RERANKER, rerank_depth=CANDIDATES)
    return [r for r in results if r["rerank_score"] >= MIN_RERANK_SCORE][:k]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    q = " ".join(sys.argv[1:]) or "lung cancer with elevated LDH and low albumin"
    hits = retrieve(q)
    if not hits:
        print(f"no passage cleared the evidence threshold ({MIN_RERANK_SCORE}) for: {q}")
    for h in hits:
        print(f"\n[{h['rerank_score']:.1f}] {h['citation']} | {h['heading'][:60]}\n  {h['text'][:240]}...")
