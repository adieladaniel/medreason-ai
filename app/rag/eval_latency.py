"""
How long does one evidence lookup take? Times the adopted retrieval pipeline (app/rag/retrieve.py:
hybrid search, MedCPT rerank of the top 5, evidence threshold) on the same 54 answerable and 16
off-corpus queries that eval_pipeline.py scores for quality.

Why this exists: the earlier notes said "about 1 to 8 seconds", which is a range from a few
sessions, not a measurement. A resume line or a Phase 7 time budget needs a median and a tail.

What is reported
  cold start   building the Retriever (loads the embedding model, scans Qdrant, builds BM25) and the
               first query (loads the MedCPT reranker). Measured once per process. The operating
               system's file cache is already warm on a rerun, so a first start after boot is slower.
  warm         per-query seconds after three untimed warm-up queries, over several passes: mean, p50,
               p95, max. Split by query style: questions ("How common is anemia in lung cancer?") and
               the terse keyword form the evidence planner actually emits ("lung cancer with low
               hemoglobin"). Off-corpus queries are timed too, since "no evidence" is a normal answer.
  rerank share the same query run without the reranker (hybrid search only), so the difference is
               what the cross-encoder costs.

Limits: answers are not cached while timing (see uncached_retrieve). One machine (CPU-only torch, 16 GB RAM, other programs running), so the numbers describe this
laptop at the moment of the run, not the software. Queries are timed one at a time, never in parallel,
which matches how the in-process worker will call them. The 54 queries are a small, fixed set.

Run:  python -m app.rag.eval_latency [--passes 3] [--out experiments/rag_retrieval/latency.json]

The result file is overwritten by each run, so give a different --out to keep a run. Speed on this laptop
swings several times with what else is running (a run with Chrome and VS Code open gave a 16 s median
where a quieter one gave 3.9 s), so close other programs first and read the first warm-up timings.
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch before sklearn/shap, see CLAUDE.md section 4)

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval" / "latency.json"
WARMUP = 3


def summarise(seconds: list[float]) -> dict:
    """Mean and percentiles in seconds. p95 of a small sample is close to the maximum, which is why the
    sample size is stored next to it."""
    a = np.asarray(seconds, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3), "min": round(float(a.min()), 3)}


def timed(fn, *args, **kwargs) -> float:
    t = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t


def uncached_retrieve(retrieve_module, query: str):
    """retrieve() keeps answers, so a repeat is free. This clears them first so every timing is a real
    lookup (hybrid search plus reranker), which is what a new query costs."""
    retrieve_module.clear_cache()
    return retrieve_module.retrieve(query)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from app.rag import retrieve as R
    from app.rag.eval_queries import OFF_CORPUS, QUERIES

    t_build = timed(R.retriever)
    first = QUERIES[0]["q"]
    t_first = timed(uncached_retrieve, R, first)
    for x in QUERIES[1:1 + WARMUP]:
        uncached_retrieve(R, x["q"])
    print(f"cold start: build {t_build:.1f} s, first query {t_first:.1f} s", flush=True)

    r = R.retriever()
    groups = {"question": [x["q"] for x in QUERIES if x["kind"] == "question"],
              "planner style": [x["q"] for x in QUERIES if x["kind"] == "case"],
              "off-corpus": list(OFF_CORPUS)}
    full: dict[str, list[float]] = {k: [] for k in groups}
    no_rerank: dict[str, list[float]] = {k: [] for k in groups}
    samples = []        # every single timing in the order it ran, so a slow stretch is visible, not averaged away
    for p in range(args.passes):
        for name, qs in groups.items():
            for i, q in enumerate(qs):
                n, f = timed(r.search, q, k=R.CANDIDATES, mode="hybrid"), timed(uncached_retrieve, R, q)
                no_rerank[name].append(n)
                full[name].append(f)
                samples.append({"pass": p + 1, "group": name, "i": i, "full": round(f, 3), "hybrid_only": round(n, 3)})
                if len(samples) % 10 == 0:
                    print(f"  {len(samples)} timings, last 10 median {np.median([x['full'] for x in samples[-10:]]):.1f} s", flush=True)
        print(f"pass {p + 1}/{args.passes} done", flush=True)

    answerable = full["question"] + full["planner style"]
    result = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "machine": {"cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
                    "gpu_used": False, "platform": platform.platform()},
        "passages": len(r.passages), "passes": args.passes, "warmup_queries": WARMUP,
        "cold_start_seconds": {"build_retriever": round(t_build, 2), "first_query": round(t_first, 2),
                               "total": round(t_build + t_first, 2)},
        "warm_seconds": {"all answerable": summarise(answerable), **{k: summarise(v) for k, v in full.items()}},
        "hybrid_search_only_seconds": summarise(sum(no_rerank.values(), [])),
        "rerank_cost_seconds": summarise([f - n for k in groups for f, n in zip(full[k], no_rerank[k])]),
        "samples": samples,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
