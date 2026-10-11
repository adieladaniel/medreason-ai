"""
Compares chunking strategies (and, jointly, embedding models) on retrieval quality, so the
default in app/rag/chunker.py and app/rag/index.py is chosen by measurement.

Two levels of "hit", because the paper-level one used in eval_retrieval.py is lenient:
  paper    a top-k passage comes from an acceptable paper
  passage  a top-k passage comes from an acceptable paper AND contains the terms the
           question is actually about (every regex group in MUST must match). This is a
           proxy for "the passage bears on the question", cheaper than human labelling but
           still only a proxy: containing the words is necessary, not sufficient.

Also reported: how many words the top-3 passages put in front of the explanation LLM
(bigger passages cost context on a 4 GB GPU), and whether a score threshold can separate
answerable from off-corpus queries (the system must be able to say "no evidence").

Honest limits: 19 answerable queries, so one query is 5 points and most gaps are noise;
the expected papers and MUST patterns were written by the project author with the
assistant; query 19 was added after the first retrieval run missed it (error analysis), so
it is not a blind test. Parent-child ("small to big") retrieval is not scored here: it
changes what the explanation writer is shown, not which passage is retrieved, so it is
judged when the explanation writer exists.

Brute-force cosine in numpy is used instead of Qdrant: identical math for a few hundred
vectors, and it keeps the comparison independent of the storage layer.

Run:  python -m app.rag.eval_chunking
"""
from __future__ import annotations

import torch  # noqa: F401

import json
import re
import sys
from pathlib import Path

import numpy as np

from app.rag.chunker import load_chunks
from app.rag.eval_retrieval import OFF_CORPUS, QUERIES
from app.rag.index import QUERY_PREFIX, _model

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval"
MODELS = ["BAAI/bge-small-en-v1.5", "NeuML/pubmedbert-base-embeddings"]

ALL_QUERIES = QUERIES   # includes query 19, added after error analysis (see module docstring)

MUST = {
    "What chest X-ray findings are typical of community-acquired pneumonia?": [r"consolidat|infiltrat|opacit|air.?bronchogram"],
    "Does a consolidation that persists after pneumonia treatment suggest lung cancer?": [r"consolidat|infiltrat|persist", r"cancer|malignan|carcinoma"],
    "How accurate is C-reactive protein for diagnosing pneumonia?": [r"C-reactive|CRP", r"sensitiv|specific|accura|AUC|area under"],
    "Does a high neutrophil-to-lymphocyte ratio predict death in pneumonia?": [r"neutrophil", r"mortality|death"],
    "What is the prognostic value of the neutrophil to lymphocyte ratio in lung cancer?": [r"NLR|PLR|lymphocyte ratio", r"surviv|prognos"],
    "Is elevated lactate dehydrogenase associated with worse survival in lung cancer?": [r"LDH|lactate dehydrogenase", r"surviv|prognos|mortality"],
    "Why are lung cancers missed on chest radiographs?": [r"miss|overlook|blind.?spot|detect"],
    "How are pulmonary nodules larger than 3 cm classified and managed?": [r"3 ?cm|30 ?mm|\bmass"],
    "lung cancer with elevated LDH and low albumin": [r"LDH|lactate dehydrogenase"],
    "pneumonia with leukocytosis and elevated CRP": [r"leukocyt|white blood|WBC|C-reactive|CRP"],
    "high neutrophil-lymphocyte ratio and platelet-lymphocyte ratio in lung cancer": [r"NLR|PLR|lymphocyte ratio"],
    "consolidation not resolving after antibiotics, possible malignancy": [r"consolidat|resol|persist", r"malignan|cancer|carcinoma"],
    "lung mass on chest radiograph with hilar enlargement": [r"hilar|\bmass"],
    "consolidation with air bronchograms on chest X-ray, pneumonia": [r"consolidat|air.?bronchogram"],
    "elevated neutrophil-lymphocyte ratio and mortality in pneumonia": [r"neutrophil", r"mortality"],
    "spiculated pulmonary nodule, lung cancer risk": [r"spicul|nodule"],
    "adenocarcinoma presenting as pneumonia": [r"pneumonia", r"adenocarcinoma"],
    "elevated procalcitonin, suspected bacterial pneumonia": [r"procalcitonin|PCT"],
    "upper lobe consolidation with elevated white blood cell count": [r"consolidat|pneumonia", r"leukocyt|white blood|WBC"],
}
assert all(q in MUST for _, q, _ in ALL_QUERIES)

STRATEGIES = {
    "sentence 100w": dict(target_words=100, overlap_words=None),
    "sentence 170w, 1-sentence overlap": dict(target_words=170, overlap_words=None),
    "sentence 300w + ~15% overlap": dict(target_words=300, overlap_words=45),
    "whole section (truncated by model)": dict(strategy="section"),
    "semantic": dict(strategy="semantic"),
}


def _encode(model_name):
    m = _model(model_name)
    return lambda texts: m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def _matches(text, groups):
    return all(re.search(g, text, re.I) for g in groups)


def evaluate(model_name: str, strategy_kwargs: dict) -> dict:
    enc = _encode(model_name)
    kw = dict(strategy_kwargs)
    if kw.get("strategy") == "semantic":
        kw["embed"] = enc
    chunks = load_chunks(**kw)
    E = enc([c.embed_text for c in chunks])
    prefix = QUERY_PREFIX.get(model_name, "")

    def top(q, k=10):
        sims = E @ enc([prefix + q])[0]
        idx = np.argsort(-sims)[:k]
        return [(chunks[i], float(sims[i])) for i in idx]

    paper_rank, passage_rank, ctx = [], [], []
    for _, q, expected in ALL_QUERIES:
        hits = top(q)
        pr = next((i + 1 for i, (c, _) in enumerate(hits) if c.pmcid in expected), None)
        ps = next((i + 1 for i, (c, _) in enumerate(hits) if c.pmcid in expected and _matches(c.text, MUST[q])), None)
        paper_rank.append(pr)
        passage_rank.append(ps)
        ctx.append(sum(len(c.text.split()) for c, _ in hits[:3]))

    def at(ranks, k):
        return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3)

    on = [top(q, 1)[0][1] for _, q, _ in ALL_QUERIES]
    off = [top(q, 1)[0][1] for q in OFF_CORPUS]
    return {
        "chunks": len(chunks), "median_words": int(np.median([len(c.text.split()) for c in chunks])),
        "paper_hit@1": at(paper_rank, 1), "paper_hit@3": at(paper_rank, 3),
        "passage_hit@1": at(passage_rank, 1), "passage_hit@3": at(passage_rank, 3), "passage_hit@5": at(passage_rank, 5),
        "passage_mrr": round(sum(1 / r for r in passage_rank if r) / len(passage_rank), 3),
        "top3_context_words": int(np.mean(ctx)),
        "threshold_gap": round(min(on) - max(off), 3),
        "missed_passage_queries": [q for (_, q, _), r in zip(ALL_QUERIES, passage_rank) if not r or r > 3],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in MODELS:
        results[m] = {}
        for name, kw in STRATEGIES.items():
            r = results[m][name] = evaluate(m, kw)
            print(f"{m.split('/')[-1][:18]:<18} {name:<36} chunks {r['chunks']:>4}  "
                  f"paper@1 {r['paper_hit@1']:.2f}  passage@1 {r['passage_hit@1']:.2f} @3 {r['passage_hit@3']:.2f} "
                  f"mrr {r['passage_mrr']:.2f}  ctx3 {r['top3_context_words']:>4}w  gap {r['threshold_gap']:+.3f}")
    (OUT_DIR / "chunking.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["# Chunking strategy comparison", "",
             f"{len(ALL_QUERIES)} answerable queries, {len(OFF_CORPUS)} off-corpus. passage = right paper AND the "
             "question's key terms present. One query is worth about 5 points, so most gaps are noise. "
             "ctx3 = words the explanation LLM would be shown for the top 3. gap = lowest answerable top score minus "
             "highest off-corpus top score (positive means a score threshold can refuse off-corpus queries).", "",
             "| model | strategy | chunks | median words | paper@1 | passage@1 | passage@3 | passage MRR | ctx3 | gap |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rows in results.items():
        for name, r in rows.items():
            lines.append(f"| {m.split('/')[-1]} | {name} | {r['chunks']} | {r['median_words']} | {r['paper_hit@1']} | "
                         f"{r['passage_hit@1']} | {r['passage_hit@3']} | {r['passage_mrr']} | {r['top3_context_words']} | "
                         f"{r['threshold_gap']:+.3f} |")
    (OUT_DIR / "CHUNKING.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
