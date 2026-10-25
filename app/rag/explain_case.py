"""
Phase 6 entry point: agent result in, verified cited explanation out.

    plan_queries -> assemble (retrieve + number passages) -> write_explanation (Ollama)
    -> verify (NLI + numbers + wording) -> text

The agent result is read-only here. The predicted class in the output is copied from the agent,
and nothing in this chain can change it: the planner and retriever only produce text queries and
passages, the writer's case sentences are built by code, and the checker only removes sentences.

CLI (runs the agent first, then this):
    python -m app.rag.explain_case --study-id 53383543 [--drop image|blood|text]
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch before sklearn/shap, see CLAUDE.md gotcha 7)

import argparse
import json
import sys

from app.rag.evidence import assemble
from app.rag.faithfulness import verify
from app.rag.planner import plan_queries
from app.rag.writer import write_explanation


def compose(plan: dict, evidence: dict, result: dict) -> dict:
    """The shown explanation from a verified result: case sentences, verified background with its [n]
    markers, one note per kind of gap, and the references actually cited. Split out of explain() so the
    staged pipeline can run planning, retrieval, writing and verifying as separate steps and still end
    with exactly this."""
    cited = sorted({b["n"] for b in result["verified"]})
    refs = [{"n": p["n"], "citation": p["citation"], "url": p["url"]} for p in evidence["passages"] if p["n"] in cited]
    notes = []
    if result["no_evidence_for"]:
        notes.append("The evidence library has no passage for: " + "; ".join(result["no_evidence_for"]) + ".")
    if result["dropped"]:
        notes.append(f"{len(result['dropped'])} drafted sentence(s) were removed because they did not match the "
                     "passage they cited.")
    if result["not_addressed"]:
        notes.append(f"{len(result['not_addressed'])} retrieved passage(s) were judged off topic and not used.")
    text = list(result["case_sentences"]) + [f"{b['text']} [{b['n']}]" for b in result["verified"]] + notes
    return {"predicted_class": result["predicted_class"], "plan": plan, "text": text, "references": refs,
            "verified": result["verified"], "dropped": result["dropped"], "pass_rate": result["pass_rate"],
            "model": result["model"]}


def explain(agent: dict, model: str | None = None) -> dict:
    plan = plan_queries(agent)
    evidence = assemble(plan)
    kw = {"model": model} if model else {}
    written = write_explanation(agent, plan, evidence, **kw)
    result = verify(written, evidence)
    return compose(plan, evidence, result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--drop", choices=["image", "blood", "text"], action="append", default=[])
    ap.add_argument("--model")
    a = ap.parse_args()
    from app.agent.orchestrator import _case_from_split, run_case
    case = _case_from_split(a.study_id, a.split, set(a.drop))
    print(f"study {a.study_id}  true label: {case['true_label']}  (the first run after a start loads the models and takes a minute or more)")
    agent = run_case(image_path=case["image_path"], labs=case["labs"], indication=case["indication"])
    out = explain(agent, a.model)
    print()
    print(f"Predicted: {out['predicted_class']} (copied from the agent, not from the writer)")
    for line in out["text"]:
        print("-", line)
    for r in out["references"]:
        print(f"[{r['n']}] {r['citation']} {r['url']}")
    print("sentences verified:", "n/a (no background written)" if out["pass_rate"] is None else f"{out['pass_rate']:.0%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
