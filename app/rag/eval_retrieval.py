"""
Scores candidate embedding models on retrieval quality, so the model in app/rag/index.py
is chosen by measurement.

Two kinds of query, because the corpus will be asked two kinds of thing:
  - question: how a clinician would phrase it
  - case: the terse keyword style the evidence planner will actually generate from a
    prediction plus its top attributions ("lung cancer, elevated LDH, low albumin")

Each query lists which papers are acceptable answers. A query counts as a hit at k if any
of the top-k passages comes from an acceptable paper. Metrics: hit@1, hit@3, hit@5, and
MRR (mean reciprocal rank of the first acceptable passage, 0 if none in the top 10).

Limits worth stating in the writeup: 19 queries is small, so a difference of one query
between two models is noise; the expected-paper lists were written by the project author
(with the assistant) from the papers' titles and abstracts, not by an independent
annotator; and a hit means the right paper surfaced, not that the passage answers the
question. This is a sanity check on model choice, not a benchmark.

Run:  python -m app.rag.eval_retrieval
"""
from __future__ import annotations

import torch  # noqa: F401

import json
import sys
from pathlib import Path

from app.rag.index import DEFAULT_MODEL, MIN_SCORE, build_index, search

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval"

MODELS = [
    "sentence-transformers/multi-qa-distilbert-cos-v1",
    "BAAI/bge-small-en-v1.5",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "NeuML/pubmedbert-base-embeddings",
]

PNEU_LAB = {"PMC1821031", "PMC4608327", "PMC10253144"}
CA_LAB = {"PMC10313305", "PMC4812293", "PMC5540491", "PMC12688816"}
CA_IMG = {"PMC8206195", "PMC9431813", "PMC6784443"}

QUERIES = [
    # clinician-style questions
    ("question", "What chest X-ray findings are typical of community-acquired pneumonia?", {"PMC3898785", "PMC10959314"}),
    ("question", "Does a consolidation that persists after pneumonia treatment suggest lung cancer?", {"PMC10959314", "PMC10035766"}),
    ("question", "How accurate is C-reactive protein for diagnosing pneumonia?", PNEU_LAB),
    ("question", "Does a high neutrophil-to-lymphocyte ratio predict death in pneumonia?", {"PMC12554325"}),
    ("question", "What is the prognostic value of the neutrophil to lymphocyte ratio in lung cancer?", {"PMC10313305", "PMC4812293"}),
    ("question", "Is elevated lactate dehydrogenase associated with worse survival in lung cancer?", {"PMC5540491", "PMC12688816"}),
    ("question", "Why are lung cancers missed on chest radiographs?", {"PMC9431813"}),
    ("question", "How are pulmonary nodules larger than 3 cm classified and managed?", {"PMC6784443", "PMC8206195"}),
    # case-style queries, as the evidence planner will phrase them
    ("case", "lung cancer with elevated LDH and low albumin", {"PMC12688816", "PMC5540491"}),
    ("case", "pneumonia with leukocytosis and elevated CRP", PNEU_LAB),
    ("case", "high neutrophil-lymphocyte ratio and platelet-lymphocyte ratio in lung cancer", {"PMC10313305", "PMC4812293"}),
    ("case", "consolidation not resolving after antibiotics, possible malignancy", {"PMC10959314", "PMC10035766"}),
    ("case", "lung mass on chest radiograph with hilar enlargement", CA_IMG),
    ("case", "consolidation with air bronchograms on chest X-ray, pneumonia", {"PMC3898785", "PMC10959314"}),
    ("case", "elevated neutrophil-lymphocyte ratio and mortality in pneumonia", {"PMC12554325"}),
    ("case", "spiculated pulmonary nodule, lung cancer risk", {"PMC6784443", "PMC8206195", "PMC9431813"}),
    ("case", "adenocarcinoma presenting as pneumonia", {"PMC10035766"}),
    ("case", "elevated procalcitonin, suspected bacterial pneumonia", {"PMC10253144", "PMC4608327"}),
    # added after error analysis: the first run returned irrelevant passages for this one, so it
    # is not a blind test
    ("case", "upper lobe consolidation with elevated white blood cell count", PNEU_LAB),
]


# medical but outside the corpus: the system should refuse to cite anything for these
OFF_CORPUS = [
    "management of diabetic ketoacidosis in adults", "recovery after total knee replacement surgery",
    "treatment of hypertension in pregnancy", "inhaler technique in pediatric asthma",
    "MRI findings in multiple sclerosis", "timing of surgery for hip fracture",
    "colorectal cancer screening colonoscopy intervals", "antidepressant dosing in elderly patients",
]


def threshold_report(model: str) -> dict:
    on = [search(q, 1, model)[0]["score"] for _, q, _ in QUERIES]
    off = [search(q, 1, model)[0]["score"] for q in OFF_CORPUS]
    return {"answerable_min": round(min(on), 3), "off_corpus_max": round(max(off), 3),
            "gap": round(min(on) - max(off), 3), "separable": min(on) > max(off)}


def score(model: str) -> dict:
    rows = []
    for kind, q, expected in QUERIES:
        ranks = [h["pmcid"] for h in search(q, k=10, model_name=model)]
        first = next((i + 1 for i, p in enumerate(ranks) if p in expected), None)
        rows.append({"kind": kind, "query": q, "first_hit_rank": first, "top3": ranks[:3]})

    def agg(sel):
        r = [x for x in rows if sel(x)]
        n = len(r)
        return {
            "n": n,
            "hit@1": round(sum(1 for x in r if x["first_hit_rank"] == 1) / n, 3),
            "hit@3": round(sum(1 for x in r if x["first_hit_rank"] and x["first_hit_rank"] <= 3) / n, 3),
            "hit@5": round(sum(1 for x in r if x["first_hit_rank"] and x["first_hit_rank"] <= 5) / n, 3),
            "mrr": round(sum(1 / x["first_hit_rank"] for x in r if x["first_hit_rank"]) / n, 3),
        }

    return {"all": agg(lambda x: True), "question": agg(lambda x: x["kind"] == "question"),
            "case": agg(lambda x: x["kind"] == "case"), "rows": rows}


def write_report(results: dict) -> None:
    n_q = sum(1 for k, _, _ in QUERIES if k == "question")
    n_c = len(QUERIES) - n_q
    lines = ["# Retrieval evaluation", "",
             f"{len(QUERIES)} answerable queries ({n_q} clinician-style questions, {n_c} case-style) and "
             f"{len(OFF_CORPUS)} off-corpus queries. A hit means the top-k passages include one from an "
             f"acceptable paper. Small sample: one query is worth {100 / len(QUERIES):.1f} points, so "
             "differences between models are mostly noise. See the docstring of "
             "app/rag/eval_retrieval.py for the stated limits.", "",
             "| model | hit@1 | hit@3 | hit@5 | MRR | lowest answerable top score | highest off-corpus top score | separable |",
             "|---|---|---|---|---|---|---|---|"]
    for m, r in results.items():
        a, t = r["all"], r["threshold"]
        lines.append(f"| {m} | {a['hit@1']} | {a['hit@3']} | {a['hit@5']} | {a['mrr']} | "
                     f"{t['answerable_min']} | {t['off_corpus_max']} | {'yes' if t['separable'] else 'no'} |")
    h1 = [r["all"]["hit@1"] for r in results.values()]
    spread = round((max(h1) - min(h1)) * len(QUERIES))
    sep = [m for m, r in results.items() if r["threshold"]["separable"]]
    widest = max(results, key=lambda m: results[m]["threshold"]["gap"])
    t = results.get(DEFAULT_MODEL, {}).get("threshold")
    inside = ""
    if t:
        inside = (f" For the default model the floor {MIN_SCORE} "
                  + ("sits inside" if t["off_corpus_max"] < MIN_SCORE < t["answerable_min"] else "is OUTSIDE")
                  + f" the gap between {t['off_corpus_max']} and {t['answerable_min']}.")
    lines += ["",
              f"Default in app/rag/index.py: `{DEFAULT_MODEL}`, minimum passage score {MIN_SCORE}.{inside}",
              f"hit@1 ranges from {min(h1)} to {max(h1)} across the models, a difference of {spread} of "
              f"{len(QUERIES)} queries, which is within noise. Models whose lowest answerable score is above "
              f"their highest off-corpus score: {', '.join(sep) if sep else 'none'}. Widest margin: {widest} "
              f"({results[widest]['threshold']['gap']:+.3f}). Every value depends on the chunking in force "
              "when this ran (see app/rag/chunker.py); experiments/rag_retrieval/CHUNKING.md compares "
              "chunking strategies."]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--report-only" in sys.argv:       # regenerate RESULTS.md from saved metrics, no re-embedding
        write_report(json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8")))
        return
    results = {}
    for m in MODELS:
        try:
            build_index(m)
            results[m] = score(m)
        except Exception as e:  # a model that fails to download or load should not sink the comparison
            print(f"SKIPPED {m}: {type(e).__name__}: {str(e)[:120]}")
            continue
        r = results[m]
        r["threshold"] = threshold_report(m)
        print(f"{m}\n   all   {r['all']}\n   case  {r['case']}\n   quest {r['question']}\n   threshold {r['threshold']}")
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
