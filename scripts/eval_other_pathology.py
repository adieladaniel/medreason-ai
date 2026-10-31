"""
What does the system do with a patient whose condition is none of its three classes?

When the dataset was built, every study that was neither Normal, Pneumonia nor Lung Cancer was labelled Excluded and dropped, so the
models were never shown "something else". This runs the unchanged system, through the app's own prediction path
(app.pipelines.staged.stage_predict: the pooling, the calibrated low-confidence rule, the prediction set and the cancer alert), on a
sample of those studies and compares how it behaves with how it behaves on the in-scope validation studies. The plan, the
definitions and what the result can and cannot say are in experiments/other_pathology/PLAN.md, written before the data was built.

The Excluded studies are those of patients who are in neither the training split nor the sealed test split
(scripts/build_final_dataset.py --excluded builds data/processed/excluded.csv), so the system has never been trained on them.

Run from the repo root, after:  python scripts/build_final_dataset.py --excluded --output data/processed/excluded.csv
    python scripts/eval_other_pathology.py [--n 1500]
Writes experiments/other_pathology/results.json (aggregates) and studies.csv (per study, git-ignored: patient identifiers).
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "other_pathology"
CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
SLUG = ["normal", "pneumonia", "lung_cancer"]
HIGH = 0.85                      # the planner's "high confidence" level (app/agent/planner.py)
B = 2000
MENTIONS = {"effusion": r"\beffusions?\b", "edema": r"\b(?:pulmonary )?(?:edema|oedema)\b", "atelectasis": r"\batelecta\w*",
            "pneumothorax": r"\bpneumothora\w*", "cardiomegaly": r"\b(?:cardiomegaly|enlarged heart|cardiac enlargement)\b",
            "fracture": r"\bfractures?\b"}


# ------------------------------------------------------------------ one record per study
def record_from_prediction(group: str, subject_id: int, pred: dict, true_label: str | None = None) -> dict:
    """pred is what stage_predict returns: the belief, the per-modality results and the decision."""
    belief, dec = pred["belief"], pred.get("decision") or {}
    p = belief["probabilities"]
    alert = dec.get("cancer_alert") or {}
    return {"group": group, "subject_id": int(subject_id), "true_label": true_label, "predicted_class": belief["predicted_class"],
            "max_prob": float(max(p.values())), "p_cancer": float(p["Lung Cancer"]),
            "modalities": "+".join(sorted(pred["modalities_used"])), "low_confidence": dec.get("low_confidence"),
            "set_size": len(dec.get("prediction_set") or []) or None,
            "alert": bool(alert.get("flag")) if alert.get("informative") else None}


def mentions(report: str) -> dict:
    text = report.lower()
    return {k: bool(re.search(rx, text)) for k, rx in MENTIONS.items()}


# ------------------------------------------------------------------ summarising (pure, tested)
def _rates(recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {}
    alert = [r["alert"] for r in recs if r["alert"] is not None]
    out = {f"names_{s}": sum(r["predicted_class"] == c for r in recs) / n for s, c in zip(SLUG, CLASSES)}
    out.update(not_low_confidence=sum(r["low_confidence"] is False for r in recs) / n,
               at_least_85=sum(r["max_prob"] >= HIGH for r in recs) / n,
               single_class_set=sum(r["set_size"] == 1 for r in recs) / n,
               alert=(sum(alert) / len(alert)) if alert else float("nan"), mean_max_prob=float(np.mean([r["max_prob"] for r in recs])))
    return out


def _stats(inside: list[dict], outside: list[dict]) -> dict:
    s = {f"in_{k}": v for k, v in _rates([r for r in inside]).items()}
    s.update({f"out_{k}": v for k, v in _rates(outside).items()})
    # the alert on the in-scope side counts only studies that are not Lung Cancer, so both sides are false alarms
    s["in_alert"] = _rates([r for r in inside if r["true_label"] != "Lung Cancer"]).get("alert", float("nan"))
    if inside and outside:
        score = -np.array([r["max_prob"] for r in inside + outside])                 # less confident = more likely out of scope
        s["auroc_confidence_detects_out_of_scope"] = float(roc_auc_score([0] * len(inside) + [1] * len(outside), score))
    return s


def summarise(records: list[dict], b: int = B, seed: int = 42) -> dict:
    inside = [r for r in records if r["group"] == "in_scope"]
    outside = [r for r in records if r["group"] == "excluded"]
    point = _stats(inside, outside)
    rng = np.random.default_rng(seed)

    def groups(recs):
        by: dict[int, list] = {}
        for r in recs:
            by.setdefault(r["subject_id"], []).append(r)
        return list(by.values())
    gi, go = groups(inside), groups(outside)
    draws = []
    for _ in range(b):
        di = [r for i in rng.integers(0, len(gi), len(gi)) for r in gi[i]] if gi else []
        do = [r for i in rng.integers(0, len(go), len(go)) for r in go[i]] if go else []
        draws.append(_stats(di, do))
    ci = {k: [float(np.nanpercentile([d[k] for d in draws if k in d], 2.5)), float(np.nanpercentile([d[k] for d in draws if k in d], 97.5))]
          for k in point if not np.isnan(point[k])}
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in point.items()}      # JSON has no NaN
    return {"n_in_scope": len(inside), "n_excluded": len(outside), "patients_in_scope": len(gi), "patients_excluded": len(go),
            "point": clean, "interval": ci,
            "excluded_modalities": pd.Series([r["modalities"] for r in outside]).value_counts().to_dict() if outside else {}}


def subgroup(records: list[dict], flags: pd.DataFrame) -> dict:
    """Descriptive rates for kinds of Excluded study. `flags` has one row per excluded record, same order."""
    out = {}
    ex = [r for r in records if r["group"] == "excluded"]
    assert len(ex) == len(flags)
    for col in flags.columns:
        idx = [i for i, v in enumerate(flags[col].values) if v]
        if idx:
            out[col] = {"n": len(idx), **{k: round(v, 3) for k, v in _rates([ex[i] for i in idx]).items() if k != "alert"}}
    return out


# ------------------------------------------------------------------ in-scope side, from stored predictions
def inscope_records(val: pd.DataFrame) -> list[dict]:
    from app.agent.orchestrator import _case_from_split
    from app.agent.decision import decide
    from app.agent.reasoning import reason
    from app.config import get_settings
    from app.pipelines.staged import provided_inputs

    cfg = get_settings()
    cols = {"image": ("image", "image_p_"), "blood": ("blood", "blood_p_"), "text": ("text", "text_indication_p_")}
    stored = {m: pd.read_csv(ROOT / "experiments" / f"{d}_baseline" / "val_predictions.csv").set_index("study_id")
              for m, (d, _) in cols.items()}
    recs = []
    for _, row in val.iterrows():
        case = _case_from_split(int(row.study_id), "val", set())
        have = provided_inputs(case["indication"], case["labs"] if case["labs"] else None, case["image_path"])
        probs = {m: {c: float(stored[m].loc[row.study_id, cols[m][1] + s]) for c, s in zip(CLASSES, SLUG)} for m in have}
        r = reason(probs)
        pred = {"belief": r["belief"], "modalities_used": list(have),
                "decision": decide(r["belief"]["probabilities"], list(have), set_coverage=cfg.prediction_set_coverage,
                                   alert_recall=cfg.cancer_alert_recall)}
        recs.append(record_from_prediction("in_scope", row.subject_id, pred, row.final_label))
    return recs


# ------------------------------------------------------------------ the Excluded side, live through the app's path
def excluded_records(sample: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    from app.agent.orchestrator import _case_from_split
    from app.pipelines.staged import provided_inputs, stage_predict

    recs, flags, t0 = [], [], time.perf_counter()
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        case = _case_from_split(int(row.study_id), "excluded", set())
        inputs = provided_inputs(case["indication"], case["labs"] if case["labs"] else None, case["image_path"])
        if not inputs:
            continue
        pred = stage_predict({"deps": {}, "inputs": inputs})
        rec = record_from_prediction("excluded", row.subject_id, pred)
        recs.append(rec)
        report = Path(row.report_path).read_text(encoding="utf-8", errors="ignore")
        flags.append({**mentions(report), "tuberculosis_coded": bool(row.icd_tb)})
        if i % 100 == 0:
            print(f"  {i}/{len(sample)} studies, {time.perf_counter() - t0:.0f} s", flush=True)
    return recs, pd.DataFrame(flags)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    excluded = pd.read_csv(ROOT / "data" / "processed" / "excluded.csv")
    labels = pd.read_csv(ROOT / "data" / "interim" / "final_study_labels.csv", usecols=["study_id", "icd_tb"])
    excluded = excluded.merge(labels, on="study_id", how="left").fillna({"icd_tb": 0})
    sample = excluded.sample(min(args.n, len(excluded)), random_state=args.seed)
    print(f"Excluded studies of unseen patients: {len(excluded)} ({excluded.subject_id.nunique()} patients); using {len(sample)}")

    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    inside = inscope_records(val)
    outside, flags = excluded_records(sample)
    records = inside + outside
    result = summarise(records)
    result["subgroups_of_excluded"] = subgroup(records, flags)
    result["seed"], result["sample_requested"] = args.seed, args.n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    pd.DataFrame(records).to_csv(OUT / "studies.csv", index=False)

    p, i = result["point"], result["interval"]

    def f(k):
        if p.get(k) is None:
            return "n/a"
        return f"{p[k]:.3f} ({i[k][0]:.3f} to {i[k][1]:.3f})" if k in i else f"{p[k]:.3f}"
    print(f"\nin scope: {result['n_in_scope']} studies, excluded: {result['n_excluded']} studies ({result['patients_excluded']} patients)")
    print(f"{'':32}{'in scope':>26}{'other pathology':>28}")
    for label, k in (("names Normal", "names_normal"), ("names Pneumonia", "names_pneumonia"), ("names Lung Cancer", "names_lung_cancer"),
                     ("not flagged low-confidence", "not_low_confidence"), ("pooled probability >= 0.85", "at_least_85"),
                     ("single-class prediction set", "single_class_set"), ("cancer alert (non-cancer only)", "alert")):
        print(f"{label:32}{f('in_' + k):>26}{f('out_' + k):>28}")
    print("\nAUROC, confidence as a detector of 'outside the three classes':", f("auroc_confidence_detects_out_of_scope"))
    for k, v in result["subgroups_of_excluded"].items():
        print(f"  {k:20} n={v['n']:4}  names Pneumonia {v['names_pneumonia']:.2f}  Cancer {v['names_lung_cancer']:.2f}  "
              f"not low-confidence {v['not_low_confidence']:.2f}  >=0.85 {v['at_least_85']:.2f}")


if __name__ == "__main__":
    main()
