"""
Blood/tabular unimodal baseline for the 3-class target (Normal / Pneumonia / Lung Cancer).

Question this answers: how much of the diagnosis is recoverable from the linked
admission's blood panel alone? This is a Phase 2 unimodal baseline -- its numbers are
the bar that multimodal fusion (Phase 3) must clear to be worth building.

Design notes:
- 17 tiered lab columns are used as-is. CRP/ESR are ~90-95% missing but are kept in:
  whether an inflammatory marker was *ordered at all* is itself clinical signal, and a
  NaN-native model can exploit that. No rows are dropped for missingness.
- Two derived features (NLR = neutrophil/lymphocyte ratio, PLR = platelet/lymphocyte
  ratio) are added -- textbook inflammation/infection markers, cheap to compute.
- Three models of increasing capability: a most-frequent DummyClassifier (floor),
  L2 logistic regression (linear reference, needs imputation), and sklearn's
  HistGradientBoostingClassifier (primary -- native NaN handling, models interactions).
- Class imbalance (Pneumonia 7312 / Normal 2953 / Cancer 774 in train) is handled with
  class_weight="balanced" throughout.
- Metric parity with the image baseline: macro F1 on the val set is the headline number.
  The test split is never read here.

Runs locally on CPU in well under a minute -- no GPU, no Kaggle.
Run from repo root:  python scripts/train_blood_baseline.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed")
OUT_DIR = Path("experiments/blood_baseline")
SEED = 42

LABEL_MAP = {"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]

BLOOD_COLS = [
    "wbc", "rbc", "hemoglobin", "hematocrit", "platelets",
    "neutrophils", "lymphocytes", "monocytes",
    "crp", "esr", "albumin", "ldh",
    "sodium", "potassium", "glucose", "creatinine", "urea",
]
CORE_COLS = ["wbc", "rbc", "hemoglobin", "hematocrit", "platelets"]
DERIVED_COLS = ["nlr", "plr"]
FEATURES = BLOOD_COLS + DERIVED_COLS


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """NLR / PLR -- standard inflammation ratios. NaN if either input is missing;
    guard against divide-by-zero when the lymphocyte percentage is recorded as 0."""
    out = df.copy()
    lymph = out["lymphocytes"].replace(0, np.nan)
    out["nlr"] = out["neutrophils"] / lymph
    out["plr"] = out["platelets"] / lymph
    return out


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df = add_derived(df)
    df["y"] = df["final_label"].map(LABEL_MAP)
    return df


def evaluate(name: str, model, X, y_true, results: dict) -> np.ndarray:
    """Print + record per-class and macro metrics for one model on one set."""
    y_pred = model.predict(X)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"\n----- {name} (val) -----")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=pred), order Normal/Pneumonia/Cancer:")
    print(cm)

    results[name] = {
        "macro_f1": round(float(macro_f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(LABEL_NAMES, per_class_f1)},
        "confusion_matrix": cm.tolist(),
    }
    return y_pred


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    print(f"train n={len(train)}  val n={len(val)}")
    print("train label counts:", train["final_label"].value_counts().to_dict())

    X_train, y_train = train[FEATURES], train["y"]
    X_val, y_val = val[FEATURES], val["y"]

    results: dict = {
        "dataset": {
            "train_n": len(train), "val_n": len(val),
            "train_label_counts": train["final_label"].value_counts().to_dict(),
            "val_label_counts": val["final_label"].value_counts().to_dict(),
        },
        "features": FEATURES,
        "seed": SEED,
    }

    # --- 1. floor: always predict the majority class ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    evaluate("dummy_most_frequent", dummy, X_val, y_val, results)

    # --- 2. logistic regression: needs imputation; add_indicator keeps missingness as signal ---
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED,
        )),
    ])
    logreg.fit(X_train, y_train)
    evaluate("logreg", logreg, X_val, y_val, results)

    # --- 3. primary: histogram gradient boosting, native NaN handling ---
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        class_weight="balanced",
        random_state=SEED,
    )
    hgb.fit(X_train, y_train)
    print(f"\nHGB stopped after {hgb.n_iter_} boosting iterations")
    hgb_pred = evaluate("hist_gradient_boosting", hgb, X_val, y_val, results)

    # --- HGB, restricted to rows that actually have >=1 core lab (isolates "no data" failures) ---
    has_core = val[CORE_COLS].notna().any(axis=1).values
    macro_f1_withdata = f1_score(y_val[has_core], hgb_pred[has_core], average="macro")
    results["hist_gradient_boosting"]["macro_f1_rows_with_core_lab"] = round(float(macro_f1_withdata), 4)
    results["hist_gradient_boosting"]["val_rows_with_core_lab"] = int(has_core.sum())
    print(f"\nHGB macro F1 on the {has_core.sum()} val rows with >=1 core lab: {macro_f1_withdata:.4f}")

    # --- permutation importance for the primary model (what the labs are worth) ---
    perm = permutation_importance(
        hgb, X_val, y_val, scoring="f1_macro", n_repeats=20, random_state=SEED, n_jobs=-1,
    )
    imp = (
        pd.DataFrame({
            "feature": FEATURES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    print("\n----- permutation importance (HGB, drop in val macro F1) -----")
    print(imp.to_string(index=False))
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # --- persist models + val predicted probabilities (for Phase 3 fusion) ---
    joblib.dump(hgb, OUT_DIR / "hgb_model.joblib")
    joblib.dump(logreg, OUT_DIR / "logreg_model.joblib")

    val_proba = hgb.predict_proba(X_val)
    val_out = val[["subject_id", "study_id", "hadm_id", "final_label"]].copy()
    for i, n in enumerate(LABEL_NAMES):
        val_out[f"blood_p_{n.replace(' ', '_').lower()}"] = val_proba[:, i]
    val_out["blood_pred"] = [LABEL_NAMES[i] for i in hgb.predict(X_val)]
    val_out.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- human-readable summary ---
    d = results
    lines = [
        "# Blood/tabular unimodal baseline -- results",
        "",
        f"Train n={d['dataset']['train_n']}, Val n={d['dataset']['val_n']}. "
        "Metric: macro F1 on val (test split untouched).",
        "",
        "| model | macro F1 | balanced acc | Normal F1 | Pneumonia F1 | Cancer F1 |",
        "|---|---|---|---|---|---|",
    ]
    for key in ["dummy_most_frequent", "logreg", "hist_gradient_boosting"]:
        m = d[key]
        pc = m["f1_per_class"]
        lines.append(
            f"| {key} | {m['macro_f1']} | {m['balanced_accuracy']} | "
            f"{pc['Normal']} | {pc['Pneumonia']} | {pc['Lung Cancer']} |"
        )
    hgb_m = d["hist_gradient_boosting"]
    lines += [
        "",
        f"HGB macro F1 on the {hgb_m['val_rows_with_core_lab']} val rows with >=1 core lab: "
        f"{hgb_m['macro_f1_rows_with_core_lab']}",
        "",
        "Top features (permutation importance): "
        + ", ".join(imp.head(6)["feature"].tolist()),
        "",
        "Artifacts: hgb_model.joblib, logreg_model.joblib, feature_importance.csv, "
        "val_predictions.csv (blood-model probabilities for Phase 3 fusion), metrics.json",
        "",
        "## Comparison to the image baseline (v3, val)",
        "Image v3: macro F1 0.50 (Normal 0.56 / Pneumonia 0.77 / Cancer 0.17).",
    ]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved everything to {OUT_DIR}/")


if __name__ == "__main__":
    main()
