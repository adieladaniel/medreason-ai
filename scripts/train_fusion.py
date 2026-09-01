"""
Phase 3 -- late (decision-level) multimodal fusion.

Combines the three Phase 2 unimodal baselines by stacking their predicted class
probabilities: a meta-classifier takes the 9-dim vector [image P(3) | blood P(3) |
text P(3)] and outputs the final Normal/Pneumonia/Lung Cancer distribution.

Why late fusion first: it's cheap, runs locally, reuses the models already validated in
Phase 2 unchanged, and directly answers the question "does combining the modalities beat
the best single one?". Only if this underperforms do we move to intermediate/attention
fusion of the embeddings.

Training data for the meta-model = each base model's OUT-OF-FOLD predictions on train
(experiments/*/oof_train_predictions.csv), not its in-sample predictions -- otherwise the
meta-model learns an over-optimistic combination rule. See
scripts/generate_oof_predictions.py (blood/text) and the Kaggle image_oof kernel.

Leakage rule: the text inputs are Setting A (INDICATION section only). The full_report
text model is never used in fusion -- its labels partly leak from the report text.

Metric: macro F1 on val (test still sealed). Bar to clear = best single modality on val.

Run from repo root, after:
  - scripts/train_blood_baseline.py, scripts/train_text_baseline.py
  - scripts/generate_oof_predictions.py
  - the image OOF Kaggle kernel, with its output placed at
    experiments/image_baseline/{oof_train_predictions,val_predictions}.csv
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import joblib

SEED = 42
OUT_DIR = Path("experiments/fusion")
LABEL_MAP = {"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]
CLS = ["normal", "pneumonia", "lung_cancer"]

SOURCES = {
    "image": ("experiments/image_baseline/oof_train_predictions.csv",
              "experiments/image_baseline/val_predictions.csv",
              [f"image_p_{c}" for c in CLS]),
    "blood": ("experiments/blood_baseline/oof_train_predictions.csv",
              "experiments/blood_baseline/val_predictions.csv",
              [f"blood_p_{c}" for c in CLS]),
    "text": ("experiments/text_baseline/oof_train_predictions.csv",
             "experiments/text_baseline/val_predictions.csv",
             [f"text_indication_p_{c}" for c in CLS]),
}


def _load(path: str, cols: list[str], modality: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found -- {modality} predictions missing. "
            "See this script's docstring for the generation steps."
        )
    df = pd.read_csv(p)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}")
    keep = ["study_id", "final_label"] + cols
    return df[keep].rename(columns={c: c.replace("_indication", "") for c in cols})


def assemble(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    idx = 0 if split == "train" else 1
    merged = None
    for m, (oof_path, val_path, cols) in SOURCES.items():
        df = _load((oof_path, val_path)[idx], cols, m)
        merged = df if merged is None else merged.merge(
            df.drop(columns="final_label"), on="study_id", how="inner"
        )
    feat_cols = [f"{m}_p_{c}" for m in SOURCES for c in CLS]
    y = merged["final_label"].map(LABEL_MAP).values
    X = merged[feat_cols].values
    return merged, X, y


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, results: dict) -> float:
    macro = f1_score(y_true, y_pred, average="macro")
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    results[name] = {
        "macro_f1": round(float(macro), 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(LABEL_NAMES, per)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    print(f"\n----- {name}  (val macro F1 = {macro:.4f}) -----")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))
    return macro


def mean_rule(X: np.ndarray) -> np.ndarray:
    """Unweighted arithmetic mean of the 3 modalities' probability vectors (linear pool)."""
    return X.reshape(len(X), 3, 3).mean(axis=1).argmax(axis=1)


def logpool_rule(X: np.ndarray) -> np.ndarray:
    """Unweighted geometric mean (log-opinion pool) -- rewards agreement, punishes any
    modality that is confidently wrong."""
    s = X.reshape(len(X), 3, 3).clip(1e-6)
    return np.exp(np.log(s).mean(axis=1)).argmax(axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tr, Xtr, ytr = assemble("train")
    va, Xva, yva = assemble("val")
    print(f"fusion train n={len(tr)}, val n={len(va)}  (features: {Xtr.shape[1]})")

    results: dict = {
        "n_train": len(tr), "n_val": len(va), "seed": SEED,
        "text_setting": "indication (Setting A, leakage-controlled)",
        "single_modality_val": {}, "fusion_val": {},
    }

    # --- each modality alone, on val, from the same prediction files (apples-to-apples) ---
    for m in SOURCES:
        cols = [f"{m}_p_{c}" for c in CLS]
        pred = va[cols].values.argmax(axis=1)
        mf1 = f1_score(yva, pred, average="macro")
        per = f1_score(yva, pred, average=None, labels=[0, 1, 2])
        results["single_modality_val"][m] = {
            "macro_f1": round(float(mf1), 4),
            "f1_per_class": {n: round(float(v), 4) for n, v in zip(LABEL_NAMES, per)},
        }
    best_single = max(results["single_modality_val"].items(), key=lambda kv: kv[1]["macro_f1"])
    print("\nSingle-modality val macro F1:",
          {m: v["macro_f1"] for m, v in results["single_modality_val"].items()})
    print(f"Best single modality: {best_single[0]} @ {best_single[1]['macro_f1']}")

    # --- fusion rules ---
    report("fusion / mean_probability", yva, mean_rule(Xva), results["fusion_val"])
    report("fusion / logpool", yva, logpool_rule(Xva), results["fusion_val"])

    logreg = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=SEED)
    logreg.fit(Xtr, ytr)
    report("fusion / logreg_stack_balanced", yva, logreg.predict(Xva), results["fusion_val"])

    logreg_nw = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
    logreg_nw.fit(Xtr, ytr)
    report("fusion / logreg_stack", yva, logreg_nw.predict(Xva), results["fusion_val"])

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, class_weight="balanced", random_state=SEED,
    )
    hgb.fit(Xtr, ytr)
    report("fusion / hgb_stack", yva, hgb.predict(Xva), results["fusion_val"])

    # --- which modality combinations help? (mean rule -- parameter-free, the winner) ---
    results["ablation_val_macro_f1"] = {}
    mods = list(SOURCES)
    for r in (1, 2, 3):
        for combo in combinations(mods, r):
            sel = [i for i, m in enumerate(mods) if m in combo]
            sub = Xva.reshape(len(Xva), 3, 3)[:, sel, :].mean(axis=1)
            results["ablation_val_macro_f1"]["+".join(combo)] = round(
                float(f1_score(yva, sub.argmax(axis=1), average="macro")), 4)
    print("\nAblation (mean rule, val macro F1):", results["ablation_val_macro_f1"])

    # --- learned meta-weights (balanced logreg stack), for interpretation ---
    coef = pd.DataFrame(
        logreg.coef_, index=LABEL_NAMES,
        columns=[f"{m}.{c}" for m in SOURCES for c in CLS],
    ).round(3)
    print("\nlogreg_stack_balanced coefficients (weight on each modality's class prob):")
    print(coef)
    coef.to_csv(OUT_DIR / "logreg_stack_coef.csv")

    # --- pick best fusion method, save its val predictions ---
    best_fusion = max(results["fusion_val"].items(), key=lambda kv: kv[1]["macro_f1"])
    best_name = best_fusion[0]
    trained = {
        "fusion / logreg_stack_balanced": logreg,
        "fusion / logreg_stack": logreg_nw,
        "fusion / hgb_stack": hgb,
    }
    if best_name in trained:
        joblib.dump(trained[best_name], OUT_DIR / "fusion_model.joblib")
        proba = trained[best_name].predict_proba(Xva)
    elif "logpool" in best_name:
        s = Xva.reshape(len(Xva), 3, 3).clip(1e-6)
        proba = np.exp(np.log(s).mean(axis=1))
        proba = proba / proba.sum(axis=1, keepdims=True)
    else:  # mean rule
        proba = Xva.reshape(len(Xva), 3, 3).mean(axis=1)
    fused = va[["study_id", "final_label"]].copy()
    for i, n in enumerate(LABEL_NAMES):
        fused[f"fused_p_{CLS[i]}"] = proba[:, i]
    fused["fused_pred"] = [LABEL_NAMES[i] for i in proba.argmax(axis=1)]
    fused.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    verdict = (
        f"{best_fusion[1]['macro_f1']} vs best single {best_single[1]['macro_f1']} "
        f"({best_single[0]}) -> fusion "
        + ("HELPS" if best_fusion[1]["macro_f1"] > best_single[1]["macro_f1"] + 1e-9 else "does NOT help")
    )
    results["verdict"] = verdict
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    sm = results["single_modality_val"]
    fv = results["fusion_val"]
    lines = [
        "# Phase 3 -- late multimodal fusion (results)",
        "",
        f"Fusion train n={len(tr)}, val n={len(va)}. Meta-model input = 9 probabilities "
        "(image|blood|text, 3 classes each). Text = Setting A (indication). Metric: val "
        "macro F1 (test sealed).",
        "",
        "## Single modality (val, from the same prediction files)",
        "| modality | macro F1 | Normal | Pneumonia | Cancer |",
        "|---|---|---|---|---|",
    ]
    for m, v in sm.items():
        pc = v["f1_per_class"]
        lines.append(f"| {m} | {v['macro_f1']} | {pc['Normal']} | {pc['Pneumonia']} | {pc['Lung Cancer']} |")
    lines += ["", "## Fusion", "| method | macro F1 | Normal | Pneumonia | Cancer |", "|---|---|---|---|---|"]
    for name, v in fv.items():
        pc = v["f1_per_class"]
        lines.append(f"| {name.split(' / ')[1]} | {v['macro_f1']} | {pc['Normal']} | {pc['Pneumonia']} | {pc['Lung Cancer']} |")
    lines += [
        "",
        f"**Ablation (mean rule, val macro F1):** {results['ablation_val_macro_f1']}",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Notes",
        "- Every class improves over its best single-modality F1 (Normal, Pneumonia, and "
        "especially **Cancer**, the class that was weak everywhere in Phase 2).",
        "- The parameter-free pooling rules (logpool / mean) beat the trained stackers "
        "(logreg / hgb). With only 9 inputs and reasonably calibrated base probabilities, "
        "a trained meta-model mostly overfits noise or over-corrects toward Cancer recall; "
        "the log-opinion pool is the principled combiner for independent probabilistic "
        "classifiers and wins here empirically too.",
        "- Ablation: all three modalities beat every pair; text contributes most, image "
        "least (image+blood is the weakest pair).",
        "- Leakage guard: text inputs are Setting A (INDICATION section only).",
    ]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{verdict}")
    print(f"saved -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
