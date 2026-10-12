"""
Compares explanation-writer setups with the faithfulness checker as the judge: which local model
writes background sentences that stay faithful to their passage, and does showing the writer the
neighbouring text (parent-child, or "small to big") help.

Items: the 24 "case" style queries (the terse keyword form the planner emits) are run through the
real retrieve(); every passage that clears the abstention threshold becomes one (topic, passage)
item. The retrieval step is cached in experiments/rag_retrieval/writer_items.json so model runs do
not touch Qdrant (embedded Qdrant locks its folder per process).

For each setup the writer sees an item and returns a sentence or NOT_ADDRESSED. Reported:
  written       share of items where the model wrote a sentence (rest were NOT_ADDRESSED)
  pass          share of written sentences that pass faithfulness.check_sentence against the
                passage that was CITED (always the original passage, never the expanded one, so the
                reader can see exactly what was claimed against)
  entail        mean entailment score of written sentences
  seconds       mean seconds per item
Parent-child: the writer is shown the cited passage plus up to NEIGHBOUR_WORDS words of the next
passage from the same paper, but the check still uses the cited passage only. If the model uses
the extra text for a claim that the cited passage does not state, the check fails it. That is the
point: expansion has to earn its keep without loosening the check.

Honest limits: the judge is the checker (validated on 35 hand-made sentences, not on these
outputs), so a model could pass by writing safe, vague sentences; the table therefore also
prints mean sentence length and a few sentences to read. Sample is ~40 items.

Run:  python -m app.rag.eval_writer --build-items      (needs Qdrant, once)
      python -m app.rag.eval_writer --models llama3.2:latest,qwen2.5:3b
"""
from __future__ import annotations

import torch  # noqa: F401

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval"
ITEMS = OUT / "writer_items.json"
NEIGHBOUR_WORDS = 150


def build_items() -> None:
    from app.rag.eval_queries import QUERIES
    from app.rag.retrieve import retrieve, retriever
    passages = retriever().passages
    items, seen = [], set()
    for q in QUERIES:
        if q["kind"] != "case":
            continue
        for h in retrieve(q["q"], k=2):
            cid = h["chunk_id"]
            if (q["q"], cid) in seen:
                continue
            seen.add((q["q"], cid))
            nxt = passages[cid + 1] if cid + 1 < len(passages) and passages[cid + 1]["pmcid"] == h["pmcid"] else None
            items.append({"topic": q["q"], "chunk_id": cid, "title": h["title"], "heading": h["heading"],
                          "text": h["text"], "neighbour": " ".join(nxt["text"].split()[:NEIGHBOUR_WORDS]) if nxt else ""})
    ITEMS.write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(len(items), "items from", len({i["topic"] for i in items}), "queries")


def run(model: str, expand: bool, items: list[dict]) -> dict:
    from app.rag import faithfulness, writer
    rows, secs = [], []
    for it in items:
        shown = {**it, "text": it["text"] + ("\n\n(next passage in the same paper) " + it["neighbour"] if expand and it["neighbour"] else "")}
        t = time.time()
        reply = writer._ask(shown, it["topic"], model, 300)
        secs.append(time.time() - t)
        sent = re.sub(r"\s+", " ", reply).strip().strip('"')
        if "NOT_ADDRESSED" in sent or not sent:
            rows.append({"topic": it["topic"], "written": False})
            continue
        r = faithfulness.check_sentence(sent, it["text"])
        rows.append({"topic": it["topic"], "written": True, "sentence": sent, **r})
    w = [r for r in rows if r["written"]]
    return {"model": model, "expand": expand, "n": len(rows),
            "written": round(len(w) / len(rows), 3),
            "pass": round(sum(r["passed"] for r in w) / len(w), 3) if w else None,
            "entail": round(statistics.mean(r["entailment"] for r in w), 3) if w else None,
            "words": round(statistics.mean(len(r["sentence"].split()) for r in w), 1) if w else None,
            "seconds": round(statistics.mean(secs), 1), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-items", action="store_true")
    ap.add_argument("--models", default="llama3.2:latest")
    ap.add_argument("--expand", action="store_true", help="also run each model with neighbour text")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.build_items:
        build_items()
        return
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    results = []
    for m in a.models.split(","):
        for expand in ([False, True] if a.expand else [False]):
            r = run(m, expand, items)
            results.append(r)
            print(f"{m:<16} expand={str(expand):<5} written {r['written']:.2f}  pass {r['pass']}  "
                  f"entail {r['entail']}  words {r['words']}  {r['seconds']}s/item", flush=True)
    path = OUT / "writer_eval.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    keep = [o for o in old if (o["model"], o["expand"]) not in {(r["model"], r["expand"]) for r in results}]
    path.write_text(json.dumps(keep + results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
