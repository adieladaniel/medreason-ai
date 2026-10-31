"""
Compares retrieval pipelines (dense, keyword, hybrid, query expansion, reranking) on the
stage-2 query set in eval_queries.py, so the production pipeline is chosen by measurement.

Protocol: variants are compared on the DEV half; the TEST half is read only to confirm the
choice. Both are reported, so a variant that wins on dev and loses on test is visible.

Metrics (passage-level, binary relevance from the proxy in eval_queries.py):
  passage@k   a relevant passage is among the top k
  MRR         mean reciprocal rank of the first relevant passage (top 10)
  nDCG@5      rewards relevant passages appearing earlier, over the top 5
  abstention  ROC AUC of "best score among the returned passages" for telling answerable
              queries (54) from off-corpus ones (16). 0.5 is chance, 1.0 is perfect. Reported
              for the dense cosine score and, where a reranker ran, for its score.
A random-ranking baseline is printed for scale, since broad queries have many relevant
passages and the metrics are lenient there.

Limits: one query is about 4 points on a half; the proxy is keyword-based; queries were
written by the project author. Run:  python -m app.rag.eval_pipeline
"""
from __future__ import annotations

import torch  # noqa: F401

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from app.rag.eval_queries import OFF_CORPUS, QUERIES
from app.rag.hybrid import Retriever

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval"
MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MEDCPT = "ncbi/MedCPT-Cross-Encoder"

VARIANTS = {
    "dense (baseline)": dict(mode="dense"),
    "keyword only (BM25)": dict(mode="lexical"),
    "dense + expansion": dict(mode="dense", expand=True),
    "hybrid": dict(mode="hybrid"),
    "hybrid + expansion": dict(mode="hybrid", expand=True),
    "hybrid + expansion + MiniLM rerank": dict(mode="hybrid", expand=True, rerank=MINILM),
    "hybrid + expansion + MedCPT rerank": dict(mode="hybrid", expand=True, rerank=MEDCPT),
    "hybrid + MedCPT rerank": dict(mode="hybrid", rerank=MEDCPT),
    "hybrid + MedCPT rerank (top 10)": dict(mode="hybrid", rerank=MEDCPT, rerank_depth=10),
    "hybrid + MedCPT rerank (top 5)": dict(mode="hybrid", rerank=MEDCPT, rerank_depth=5),
}


def is_relevant(p: dict, x: dict) -> bool:
    return p["topic"] in x["topics"] and all(re.search(g, p["text"], re.I) for g in x["must"])


def metrics(rels: list[list[bool]]) -> dict:
    n = len(rels)
    def hit(k): return sum(any(r[:k]) for r in rels) / n
    def rr(r): return next((1 / (i + 1) for i, v in enumerate(r[:10]) if v), 0.0)
    def ndcg(r):
        dcg = sum(v / math.log2(i + 2) for i, v in enumerate(r[:5]))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(5, sum(r))))
        return dcg / ideal if ideal else 0.0
    return {"n": n, "p@1": round(hit(1), 3), "p@3": round(hit(3), 3), "p@5": round(hit(5), 3),
            "mrr": round(float(np.mean([rr(r) for r in rels])), 3),
            "ndcg5": round(float(np.mean([ndcg(r) for r in rels])), 3)}


def random_baseline(R: Retriever) -> dict:
    N = len(R.passages)
    out = {"p@1": [], "p@3": [], "p@5": []}
    for x in QUERIES:
        g = sum(is_relevant(p, x) for p in R.passages)
        for k in (1, 3, 5):
            out[f"p@{k}"].append(1 - np.prod([(N - g - i) / (N - i) for i in range(k)]) if g else 0.0)
    return {k: round(float(np.mean(v)), 3) for k, v in out.items()}


def run(R: Retriever, kw: dict) -> dict:
    rows, on_dense, on_rerank = [], [], []
    for x in QUERIES:
        res = R.search(x["q"], k=10, **kw)
        rows.append({"split": x["split"], "rel": [is_relevant(p, x) for p in res]})
        top5 = res[:5]
        on_dense.append(max(p["dense_score"] for p in top5))
        on_rerank.append(max((p["rerank_score"] for p in res if p["rerank_score"] is not None), default=None))
    off_dense, off_rerank = [], []
    for q in OFF_CORPUS:
        res = R.search(q, k=10, **kw)
        off_dense.append(max(p["dense_score"] for p in res[:5]))
        off_rerank.append(max((p["rerank_score"] for p in res if p["rerank_score"] is not None), default=None))

    def auc(on, off):
        if any(v is None for v in on + off):
            return None
        return round(float(roc_auc_score([1] * len(on) + [0] * len(off), on + off)), 3)

    return {
        "dev": metrics([r["rel"] for r in rows if r["split"] == "dev"]),
        "test": metrics([r["rel"] for r in rows if r["split"] == "test"]),
        "all": metrics([r["rel"] for r in rows]),
        "abstain_auc_dense": auc(on_dense, off_dense),
        "abstain_auc_rerank": auc(on_rerank, off_rerank),
        "scores": {"on_dense": on_dense, "off_dense": off_dense, "on_rerank": on_rerank, "off_rerank": off_rerank},
    }


def write_report(results: dict) -> None:
    """Markdown summary computed from the saved metrics, so it cannot disagree with them."""
    rnd = results["random"]
    lines = ["# Retrieval pipeline comparison (44 papers, 811 passages)", "",
             f"{len(QUERIES)} answerable queries split by topic group into dev and test, plus {len(OFF_CORPUS)} off-corpus "
             "queries. Passage relevance is a keyword proxy (see eval_queries.py), audited by hand in "
             "experiments/rag_retrieval/AUDIT.md. One query is about 4 points on a half, so gaps of one or two "
             f"queries are noise. Random ranking would score p@1 {rnd['p@1']}, p@3 {rnd['p@3']}, p@5 {rnd['p@5']}.", "",
             "| variant | dev p@1 | dev p@3 | dev MRR | dev nDCG@5 | test p@1 | test p@3 | test MRR | test nDCG@5 | abstention AUC (dense score) | abstention AUC (reranker score) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in VARIANTS:
        r = results.get(name)
        if not r:
            continue
        d, t = r["dev"], r["test"]
        f = lambda v: "-" if v is None else v
        lines.append(f"| {name} | {d['p@1']} | {d['p@3']} | {d['mrr']} | {d['ndcg5']} | {t['p@1']} | {t['p@3']} | "
                     f"{t['mrr']} | {t['ndcg5']} | {f(r['abstain_auc_dense'])} | {f(r['abstain_auc_rerank'])} |")
    (OUT_DIR / "PIPELINE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--report-only" in sys.argv:
        write_report(json.loads((OUT_DIR / "pipeline.json").read_text(encoding="utf-8")))
        return
    R = Retriever()
    base = random_baseline(R)
    print(f"{len(R.passages)} passages | random ranking: {base}\n")
    path = OUT_DIR / "pipeline.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    results["random"] = base
    only = None
    if "--only" in sys.argv:       # rerun just these variants (comma-separated), keep the rest
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    print(f"{'variant':<40}{'dev p@1':>8}{'p@3':>6}{'mrr':>6}{'ndcg5':>7} |{'test p@1':>9}{'p@3':>6}{'mrr':>6}{'ndcg5':>7} |{'AUC dense':>10}{'AUC rerank':>11}")
    for name, kw in VARIANTS.items():
        if only is not None and name not in only:
            continue
        r = results[name] = run(R, kw)
        d, t = r["dev"], r["test"]
        print(f"{name:<40}{d['p@1']:>8.2f}{d['p@3']:>6.2f}{d['mrr']:>6.2f}{d['ndcg5']:>7.2f} |{t['p@1']:>9.2f}{t['p@3']:>6.2f}{t['mrr']:>6.2f}{t['ndcg5']:>7.2f} |"
              f"{r['abstain_auc_dense'] if r['abstain_auc_dense'] is not None else '-':>10}{r['abstain_auc_rerank'] if r['abstain_auc_rerank'] is not None else '-':>11}")
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
