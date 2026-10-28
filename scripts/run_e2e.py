"""
Phase 7 step 7: run a spread of real validation cases through the running service and write down what happens.

Why this exists: every piece of the service was tested alone, on one or two cases. This answers the question the
tests cannot: across the situations the system claims to handle, does the deployed pipeline give the answer the offline
evaluation gives, and how long does each stage take? It talks to the server over HTTP exactly as the Analyze page does,
so it works against `uvicorn` and against the docker compose stack alike.

The cases are chosen by rule, with a fixed seed, so the run can be repeated and nobody can say they were picked to look good:
cases the models get right, cases they get wrong (including the cancers they miss), a case with no labs on record, and
cases with one input taken away. Only the validation split is used. The test split stays sealed (CLAUDE.md section 6).

Each case is checked against its offline twin: the stored per-modality probabilities (experiments/*_baseline/val_predictions.csv)
pooled with the app's own rule for the inputs that were actually sent. A difference means the deployed pipeline is not
reproducing the evaluated models, which is what this run is for finding.

Outputs, in experiments/e2e/:
  <out>/cases.json    one record per case, with study ids and true labels (git-ignored: MIMIC identifiers)
  <out>/summary.json  aggregates only, no identifiers (committed)

Run from the repo root with the service and Ollama up:
    python scripts/run_e2e.py --server http://127.0.0.1:8000 --passes 2
Every case is deleted from the server after it is read, because it holds a real X-ray.
"""
from __future__ import annotations

import torch  # noqa: F401  (before sklearn and shap on Windows, see CLAUDE.md gotcha 7)

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.health import percentile  # noqa: E402

OUT_DIR = ROOT / "experiments" / "e2e"
CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
SLUG = {"Normal": "normal", "Pneumonia": "pneumonia", "Lung Cancer": "lung_cancer"}
STAGES = ["predicting", "explaining", "retrieving", "writing", "verifying"]
TERMINAL = ("done", "partial", "failed")


# ------------------------------------------------------------------ choosing the cases
def select_cases(df: pd.DataFrame, lab_columns: list[str], seed: int = 42) -> list[dict]:
    """df has study_id, final_label, fused_pred (the stored fusion prediction) and the lab columns, one row per validation
    study. Returns [{study_id, category, drop}], in a fixed order, by rule and seed. A study is used at most once."""
    d = df.copy()
    d["has_labs"] = d[lab_columns].notna().any(axis=1)
    right = d["final_label"] == d["fused_pred"]
    pools = [
        ("agree, pneumonia", 2, (d.final_label == "Pneumonia") & right & d.has_labs, []),
        ("agree, normal", 2, (d.final_label == "Normal") & right & d.has_labs, []),
        ("cancer found by fusion", 2, (d.final_label == "Lung Cancer") & right & d.has_labs, []),
        ("cancer missed by fusion", 2, (d.final_label == "Lung Cancer") & ~right & d.has_labs, []),
        ("wrong, not cancer", 2, (d.final_label != "Lung Cancer") & ~right & d.has_labs, []),
        ("no labs on record", 2, ~d.has_labs, []),
        ("text only", 1, (d.final_label == "Pneumonia") & right & d.has_labs, ["image", "blood"]),
        ("image only", 1, (d.final_label == "Pneumonia") & right & d.has_labs, ["blood", "text"]),
        ("no X-ray", 1, (d.final_label == "Pneumonia") & right & d.has_labs, ["image"]),
        ("no indication", 1, (d.final_label == "Pneumonia") & right & d.has_labs, ["text"]),
    ]
    used: set[int] = set()
    chosen = []
    for k, (name, n, mask, drop) in enumerate(pools):
        pool = d[mask & ~d.study_id.isin(used)].sort_values("study_id")
        take = pool.sample(min(n, len(pool)), random_state=seed + k)
        for sid in take.study_id:
            used.add(int(sid))
            chosen.append({"study_id": int(sid), "category": name, "drop": list(drop)})
    return chosen


def expected_for(modalities_sent: set[str], study_id: int, stored: dict[str, pd.DataFrame]) -> dict:
    """What the offline evaluation says for this study and these inputs: the stored per-modality probabilities, pooled
    with the app's own rule (app.agent.reasoning.reason, the Phase 3 geometric mean)."""
    from app.agent.reasoning import reason
    probs = {}
    for m in modalities_sent:
        row = stored[m].loc[stored[m].study_id == study_id].iloc[0]
        cols = {"image": "image_p_", "blood": "blood_p_", "text": "text_indication_p_"}[m]
        probs[m] = {c: float(row[cols + SLUG[c]]) for c in CLASSES}
    belief = reason(probs)["belief"]
    return {"predicted_class": belief["predicted_class"], "probabilities": belief["probabilities"]}


# ------------------------------------------------------------------ one case
def run_one(client, server: str, spec: dict, stored: dict[str, pd.DataFrame], poll: float = 0.5) -> dict:
    from app.agent.orchestrator import _case_from_split
    from submit_case import build_request

    case = _case_from_split(spec["study_id"], "val", set())
    data, files = build_request(case, set(spec["drop"]))
    sent = ({"text"} if "indication" in data else set()) | ({"blood"} if "labs" in data else set()) | \
           ({"image"} if files else set())
    rec = {**spec, "true_label": case["true_label"], "inputs_sent": sorted(sent)}

    t0 = time.perf_counter()
    r = client.post(f"{server}/api/cases", data=data, files=files, timeout=60)
    if r.status_code != 202:
        return {**rec, "status": "rejected", "http_status": r.status_code, "error": r.text[:200]}
    cid = r.json()["id"]
    rec["time_to_prediction"] = None
    view = None
    try:
        while True:
            view = client.get(f"{server}/api/cases/{cid}", timeout=30).json()
            if rec["time_to_prediction"] is None and view.get("prediction"):
                rec["time_to_prediction"] = round(time.perf_counter() - t0, 2)
            if view["status"] in TERMINAL:
                break
            time.sleep(poll)
        rec["wall_seconds"] = round(time.perf_counter() - t0, 2)
        calls = client.get(f"{server}/api/cases/{cid}/llm-calls", timeout=30).json()["calls"]
    finally:
        client.delete(f"{server}/api/cases/{cid}", timeout=30)       # it holds a real X-ray

    rec.update(status=view["status"], error=view.get("error"), timings=view.get("timings") or {})
    pred = view.get("prediction") or {}
    belief = pred.get("belief") or {}
    if belief:
        exp = expected_for(sent, spec["study_id"], stored)
        diff = max(abs(belief["probabilities"][c] - exp["probabilities"][c]) for c in CLASSES)
        dec = pred.get("decision") or {}
        rec.update(predicted_class=belief["predicted_class"], probabilities=belief["probabilities"],
                   expected_class=exp["predicted_class"], expected_probabilities=exp["probabilities"],
                   class_matches_offline=belief["predicted_class"] == exp["predicted_class"],
                   max_probability_difference=round(diff, 5),
                   consistency=(pred.get("consistency") or {}).get("status"),
                   low_confidence=dec.get("low_confidence"), prediction_set=dec.get("prediction_set"),
                   cancer_alert=(dec.get("cancer_alert") or {}).get("flag"),
                   correct=belief["predicted_class"] == case["true_label"])
    expl = view.get("explanations") or {}
    rec["explainers_match_stored_prediction"] = all(v.get("matches_prediction") for v in expl.values()) if expl else None
    ev = view.get("evidence") or {}
    final = ev.get("final") or {}
    if ev:
        rec.update(n_passages=len(ev.get("passages") or []), n_queries_without_passage=len(ev.get("unanswered") or []),
                   n_verified=len(final.get("verified") or []), n_dropped=len(final.get("dropped") or []),
                   pass_rate=final.get("pass_rate"))
    rec["llm_call_seconds"] = [round(c["seconds"], 2) for c in calls]
    return rec


# ------------------------------------------------------------------ summarising
def _spread(values: list[float]) -> dict:
    return {"n": len(values), "p50": round(percentile(values, 0.5), 2), "p95": round(percentile(values, 0.95), 2),
            "max": round(max(values), 2)} if values else {"n": 0}


def summarise(records: list[dict]) -> dict:
    """Aggregates only, no identifiers. `records` may span several passes; each carries its own 'pass'."""
    ok = [r for r in records if r.get("status") in TERMINAL]
    done = [r for r in ok if r["status"] == "done"]
    with_pred = [r for r in ok if "predicted_class" in r]
    stage_seconds = {s: _spread([r["timings"][s] for r in ok if s in (r.get("timings") or {})]) for s in STAGES}
    drafted = [(r["n_verified"] + r["n_dropped"], r["timings"].get("verifying")) for r in done
               if r.get("n_verified") is not None and r.get("timings", {}).get("verifying") is not None]
    with_sentences = [(n, s) for n, s in drafted if n > 0]
    by_category: dict[str, dict] = {}
    for r in with_pred:
        c = by_category.setdefault(r["category"], {"n": 0, "correct": 0, "matches_offline": 0})
        c["n"] += 1
        c["correct"] += int(bool(r.get("correct")))
        c["matches_offline"] += int(bool(r.get("class_matches_offline")))
    return {
        "cases_run": len(records), "statuses": {s: sum(1 for r in records if r.get("status") == s)
                                                 for s in sorted({r.get("status") for r in records})},
        "stage_seconds": stage_seconds,
        "wall_seconds": _spread([r["wall_seconds"] for r in ok]),
        "time_to_prediction": _spread([r["time_to_prediction"] for r in ok if r.get("time_to_prediction") is not None]),
        "same_class_as_offline": {"n": len(with_pred), "same": sum(1 for r in with_pred if r["class_matches_offline"])},
        "max_probability_difference_vs_offline": _spread([r["max_probability_difference"] for r in with_pred]),
        "explainers_match_stored_prediction": {"n": sum(1 for r in ok if r.get("explainers_match_stored_prediction") is not None),
                                               "all_match": sum(1 for r in ok if r.get("explainers_match_stored_prediction"))},
        # The cases are chosen on purpose (some the fusion gets right, some wrong), so this is NOT an accuracy.
        "agrees_with_true_label_in_this_selected_set": {"n": len(with_pred), "agree": sum(1 for r in with_pred if r.get("correct"))},
        "evidence": {
            "cases_with_evidence": sum(1 for r in done if r.get("n_passages") is not None),
            "cases_with_a_verified_sentence": sum(1 for r in done if (r.get("n_verified") or 0) > 0),
            "verified_sentences": sum(r.get("n_verified") or 0 for r in done),
            "dropped_sentences": sum(r.get("n_dropped") or 0 for r in done),
        },
        "verifying_seconds_per_drafted_sentence": _spread([s / n for n, s in with_sentences]),
        "language_model_call_seconds": _spread([x for r in done for x in r.get("llm_call_seconds", [])]),
        "by_category": by_category,
    }


# ------------------------------------------------------------------ main
def main() -> None:
    import httpx

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--passes", type=int, default=2, help="run the whole set this many times (the first is the cold one)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    from explain.blood import RAW_LABS
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    fused = pd.read_csv(ROOT / "experiments" / "fusion" / "val_predictions.csv")[["study_id", "fused_pred"]]
    specs = select_cases(val.merge(fused, on="study_id"), list(RAW_LABS), args.seed)
    stored = {m: pd.read_csv(ROOT / "experiments" / f"{m}_baseline" / "val_predictions.csv") for m in ("image", "blood", "text")}

    with httpx.Client() as client:
        ready = client.get(f"{args.server}/api/ready", timeout=30).json()
        print("ready:", ready["status"], {k: v["ok"] for k, v in ready["checks"].items()})
        if ready["status"] != "ok":
            sys.exit("the service is not fully ready; start Ollama and wait for the models to load")
        records = []
        for p in range(1, args.passes + 1):
            for i, spec in enumerate(specs, 1):
                rec = {**run_one(client, args.server, spec, stored), "pass": p, "order": i}
                records.append(rec)
                print(f"pass {p} case {i:2}/{len(specs)} {spec['category']:<24} {rec['status']:<8} "
                      f"{rec.get('wall_seconds', '-')} s  class={rec.get('predicted_class')} "
                      f"offline={rec.get('expected_class')} diff={rec.get('max_probability_difference')}", flush=True)
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                (out / "cases.json").write_text(json.dumps(records, indent=1), encoding="utf-8")   # kept as it goes

    out = Path(args.out)
    summary = {"seed": args.seed, "passes": args.passes, "cases_per_pass": len(specs),
               "categories": sorted({s["category"] for s in specs}), "all_passes": summarise(records),
               "by_pass": {str(p): summarise([r for r in records if r["pass"] == p]) for p in range(1, args.passes + 1)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
