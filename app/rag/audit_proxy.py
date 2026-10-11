"""
Prints a sample of (query, top-1 passage) pairs so a human can judge them, to estimate how
well the keyword proxy in eval_queries.py agrees with real relevance.

Sample: every query whose top-1 passage the proxy calls NOT relevant (can the proxy be wrong
in the strict direction?), plus a seeded random sample of queries whose top-1 the proxy calls
relevant (does "contains the terms" really mean "answers the question"?). The proxy label is
printed, so judge with it in view, but read the passage against the query, not the label.

Run after the index is built and while no other process holds the Qdrant folder:
    python -m app.rag.audit_proxy
"""
from __future__ import annotations

import torch  # noqa: F401

import random
import sys

from app.rag.eval_pipeline import MEDCPT, is_relevant
from app.rag.eval_queries import QUERIES
from app.rag.hybrid import Retriever

N_RELEVANT_SAMPLE = 12
SEED = 7


def main() -> None:
    R = Retriever()
    rows = []
    for i, x in enumerate(QUERIES):
        top = R.search(x["q"], k=1, mode="hybrid", rerank=MEDCPT)[0]
        rows.append((i, x, top, is_relevant(top, x)))
    miss = [r for r in rows if not r[3]]
    hit = [r for r in rows if r[3]]
    random.Random(SEED).shuffle(hit)
    sample = miss + hit[:N_RELEVANT_SAMPLE]
    print(f"top-1 proxy-relevant: {len(hit)}/{len(rows)}; auditing {len(miss)} proxy-misses + {min(N_RELEVANT_SAMPLE, len(hit))} proxy-hits\n")
    for n, (i, x, top, ok) in enumerate(sample, 1):
        words = top["text"].split()
        print(f"[{n}] PROXY SAYS {'RELEVANT' if ok else 'NOT relevant'} | Q: {x['q']}")
        print(f"    {top['citation']} | {top['heading'][:60]} | rerank {top['rerank_score']:.2f}")
        print(f"    {' '.join(words[:105])}{' ...' if len(words) > 105 else ''}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
