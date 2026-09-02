"""
SHAP feature attribution for the blood/tabular model.

Loads experiments/blood_baseline/hgb_model.joblib (HistGradientBoostingClassifier) and
explains one case: which lab values pushed the predicted class's log-odds up or down.

TreeExplainer in tree_path_dependent mode gives exact SHAP values for this model with no
background dataset needed; the reconstruction softmax(base + sum(shap)) == predict_proba
holds exactly. SHAP values are in log-odds (margin) space.

The 19 model features = the 17 raw labs + NLR + PLR (neutrophil/lymphocyte and
platelet/lymphocyte ratios). Feature construction mirrors scripts/train_blood_baseline.py
and is kept in sync by hand -- training code is intentionally not imported here.

CLI:  python -m explain.blood --study-id 50296395   (pulls that row's labs from val.csv)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch  # noqa: F401  -- must precede sklearn/shap import on Windows (see CLAUDE.md §4)
import numpy as np
import pandas as pd
import joblib
import shap

from explain import LABEL_NAMES

MODEL_PATH = Path("experiments/blood_baseline/hgb_model.joblib")
RAW_LABS = [
    "wbc", "rbc", "hemoglobin", "hematocrit", "platelets", "neutrophils", "lymphocytes",
    "monocytes", "crp", "esr", "albumin", "ldh", "sodium", "potassium", "glucose",
    "creatinine", "urea",
]
FEATURES = RAW_LABS + ["nlr", "plr"]
_PRETTY = {
    "wbc": "WBC", "rbc": "RBC", "hemoglobin": "hemoglobin", "hematocrit": "hematocrit",
    "platelets": "platelets", "neutrophils": "neutrophils %", "lymphocytes": "lymphocytes %",
    "monocytes": "monocytes %", "crp": "CRP", "esr": "ESR", "albumin": "albumin", "ldh": "LDH",
    "sodium": "sodium", "potassium": "potassium", "glucose": "glucose", "creatinine": "creatinine",
    "urea": "urea", "nlr": "neutrophil-lymphocyte ratio", "plr": "platelet-lymphocyte ratio",
}

_cache: dict = {}


def _feature_vector(labs: dict) -> np.ndarray:
    """17 raw lab values (missing -> nan) -> 19-vector with NLR, PLR appended."""
    d = {c: (labs.get(c) if labs.get(c) is not None else np.nan) for c in RAW_LABS}
    lymph = d["lymphocytes"]
    lymph = np.nan if (pd.isna(lymph) or lymph == 0) else lymph
    nlr = d["neutrophils"] / lymph if not pd.isna(d["neutrophils"]) and not pd.isna(lymph) else np.nan
    plr = d["platelets"] / lymph if not pd.isna(d["platelets"]) and not pd.isna(lymph) else np.nan
    return np.array([[*(d[c] for c in RAW_LABS), nlr, plr]], dtype=float)


def _load():
    if not _cache:
        model = joblib.load(MODEL_PATH)
        _cache["model"] = model
        _cache["explainer"] = shap.TreeExplainer(model, feature_names=FEATURES)
    return _cache["model"], _cache["explainer"]


def explain_blood(labs: dict, top_k: int = 8) -> dict:
    """labs: {raw_lab_name: value|None}. Returns the common attribution contract."""
    model, explainer = _load()
    x = _feature_vector(labs)
    proba = model.predict_proba(pd.DataFrame(x, columns=FEATURES))[0]
    pred = int(proba.argmax())

    exp = explainer(x, check_additivity=False)
    shap_pred = np.asarray(exp.values)[0, :, pred]          # [19] contribution to pred class log-odds

    order = np.argsort(-np.abs(shap_pred))
    items = []
    for j in order[:top_k]:
        val = x[0, j]
        items.append({
            "feature": _PRETTY[FEATURES[j]],
            "value": None if np.isnan(val) else round(float(val), 2),
            "effect": round(float(shap_pred[j]), 4),
            "direction": "supports" if shap_pred[j] > 0 else "against",
        })

    supports = [it["feature"] for it in items if it["direction"] == "supports"][:3]
    against = [it["feature"] for it in items if it["direction"] == "against"][:2]
    summary = (
        f"Blood model predicts {LABEL_NAMES[pred]} (p={proba[pred]:.2f}). "
        + (f"Pushed toward by: {', '.join(supports)}. " if supports else "")
        + (f"Pushed away by: {', '.join(against)}." if against else "")
    ).strip()

    return {
        "modality": "blood",
        "predicted_class": LABEL_NAMES[pred],
        "probabilities": {n: round(float(p), 4) for n, p in zip(LABEL_NAMES, proba)},
        "attributions": {"method": "SHAP (TreeExplainer, log-odds)", "items": items},
        "summary": summary,
        "_raw": {"shap_all_classes": np.asarray(exp.values)[0].tolist(),
                 "base_values": np.asarray(exp.base_values)[0].tolist()},
    }


def plot_blood(res: dict, out_path: str | Path) -> Path:
    """Horizontal bar of the top signed SHAP contributions for the predicted class.
    `res` is the dict returned by explain_blood()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(reversed(res["attributions"]["items"]))
    labels = [f"{it['feature']}" + (f" = {it['value']}" if it["value"] is not None else " (missing)")
              for it in items]
    effects = [it["effect"] for it in items]
    colors = ["#c0392b" if e > 0 else "#2c7fb8" for e in effects]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(effects)), effects, color=colors)
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel(f"SHAP contribution to log-odds of {res['predicted_class']}")
    ax.set_title(res["summary"], fontsize=9, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _labs_from_split(study_id: int, split: str = "val") -> dict:
    df = pd.read_csv(f"data/processed/{split}.csv")
    row = df.loc[df["study_id"] == study_id]
    if row.empty:
        raise SystemExit(f"study_id {study_id} not in {split}.csv")
    r = row.iloc[0]
    return {c: (None if pd.isna(r[c]) else float(r[c])) for c in RAW_LABS}, r["final_label"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--plot", default=None, help="path to save a bar plot PNG")
    args = ap.parse_args()

    labs, true_label = _labs_from_split(args.study_id, args.split)
    res = explain_blood(labs)
    print(f"study {args.study_id}  true label: {true_label}")
    print(f"probabilities: {res['probabilities']}")
    print(f"\n{res['summary']}\n")
    print(f"{'feature':<30}{'value':>10}{'effect':>10}  direction")
    for it in res["attributions"]["items"]:
        print(f"{it['feature']:<30}{str(it['value']):>10}{it['effect']:>10}  {it['direction']}")
    if args.plot:
        print(f"\nsaved plot -> {plot_blood(res, args.plot)}")
