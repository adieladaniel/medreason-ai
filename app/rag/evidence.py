"""
Evidence assembly: runs the planner's queries through retrieve() and returns one numbered list
of passages, each tagged with the driver it was fetched for.

Why a separate step: the writer must be able to cite "[2]" and the checker must be able to look
[2] up again, so numbering and de-duplication happen once, here, and both read the same list.

Limits chosen for a 3B model on a 4 GB GPU: at most PER_QUERY passages per query and MAX_PASSAGES
in total, so the prompt stays near 1,500 words. A passage retrieved for two drivers keeps one
number and lists both drivers. A query that retrieves nothing above the abstention threshold is
recorded under `unanswered`, so the explanation can say "the library has no passage on X"
instead of quietly leaving it out.

Retrieval only reads the planner's text. It never sees the prediction's probabilities, so it
cannot change the label.
"""
from __future__ import annotations

PER_QUERY = 2
MAX_PASSAGES = 5


def assemble(plan: dict) -> dict:
    from app.rag.retrieve import retrieve

    passages: list[dict] = []
    by_id: dict[str, dict] = {}
    unanswered: list[dict] = []
    for q in plan["queries"]:
        hits = retrieve(q["text"], k=PER_QUERY)
        if not hits:
            unanswered.append(q)
            continue
        for h in hits:
            p = by_id.get(h["chunk_id"])
            if p is None:
                if len(passages) >= MAX_PASSAGES:
                    continue
                p = {"n": len(passages) + 1, "chunk_id": h["chunk_id"], "text": h["text"],
                     "heading": h.get("heading"), "citation": h.get("citation"), "title": h.get("title"),
                     "url": h.get("url"), "rerank_score": h["rerank_score"], "drivers": [], "queries": [], "terms": set()}
                passages.append(p)
                by_id[h["chunk_id"]] = p
            if q["driver"] not in p["drivers"]:
                p["drivers"].append(q["driver"])
                p["queries"].append(q["text"])
            # a passage fetched for an ungated query (terms None) is never gated: it could be on topic for it
            p["terms"] = None if (p["terms"] is None or not q.get("terms")) else p["terms"] | set(q["terms"])
    return {"passages": passages, "unanswered": unanswered}
