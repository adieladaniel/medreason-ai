"""
Where does the evidence library have nothing to say? Runs every query the planner can emit (each
blood phrase and each image channel, for both disease classes) through the real retrieval
pipeline and reports the ones with no ON-TOPIC passage.

Why this and not the 54-query evaluation: eval_pipeline.py measures whether retrieval ranks the
right passage first for questions someone wrote. This measures coverage of the questions the system
itself will ask, which is what decides whether an explanation gets any literature at all.

A hit counts as on topic only if its text contains one of the finding's TERMS from planner.py,
the same test the writer applies before keeping a sentence. Image queries use the planner's
fallback phrasing when the primary query returns nothing, exactly as evidence.assemble does.

Some gaps are correct answers: no literature links low hematocrit to pneumonia, so "no passage"
is the honest result there. The list is for choosing which gaps are worth filling with papers,
and for checking that adding papers fills gaps without opening new ones.

History (52 queries, queries with no on-topic passage): 27 papers 29, 44 papers 15, 44 papers
with the fallback phrasing and broader opacity terms 13. Snapshots are kept in
experiments/rag_retrieval/gap_probe_*.json.

Run:  python -m app.rag.eval_gaps
"""
from __future__ import annotations

import torch  # noqa: F401

import json
import sys
from pathlib import Path

from app.rag.planner import BLOOD_RANGES, CLASS_PHRASE, IMAGE_PHRASES, TERMS

OUT = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval" / "gap_probe.json"


def planner_queries() -> list[dict]:
    out = []
    for cls in CLASS_PHRASE.values():
        for _, _, lo, hi in BLOOD_RANGES.values():
            for ph in (lo, hi):
                if ph:
                    out.append({"finding": ph, "text": f"{cls} with {ph}", "fallback": None})
        for ph in IMAGE_PHRASES.values():
            out.append({"finding": ph, "text": f"{cls} chest X-ray with {ph}", "fallback": f"{ph} versus {cls} chest radiograph"})
    seen, uniq = set(), []
    for q in out:
        if q["text"] not in seen:
            seen.add(q["text"])
            uniq.append(q)
    return uniq


def probe(queries: list[dict], retrieve) -> list[dict]:
    rows = []
    for q in queries:
        hits = retrieve(q["text"], k=2)
        if not hits and q["fallback"]:
            hits = retrieve(q["fallback"], k=2)
        terms = TERMS.get(q["finding"]) or ()
        on = [h for h in hits if any(t in h["text"].lower() for t in terms)]
        rows.append({"query": q["text"], "hits": len(hits), "on_topic_hits": len(on),
                     "best": round(hits[0]["rerank_score"], 1) if hits else None})
    return rows


def main() -> None:
    from app.rag.retrieve import retrieve
    rows = probe(planner_queries(), retrieve)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    gaps = [r for r in rows if r["on_topic_hits"] == 0]
    print(f"{len(rows)} planner queries, {len(gaps)} with no on-topic passage")
    for r in gaps:
        print("  GAP", r["query"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
