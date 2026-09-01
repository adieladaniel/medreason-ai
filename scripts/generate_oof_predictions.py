"""
Out-of-fold (OOF) train predictions for the blood and text (indication) baselines --
the piece Phase 3 fusion needs that val_predictions.csv can't provide.

WHY THIS EXISTS
----------------
The fusion meta-model is trained on the base models' predicted probabilities. If we fed
it each base model's predictions on the *same rows it was trained on*, those predictions
would be overconfident/overfit and the meta-model would learn a distorted (too-easy)
combination rule -- it would look great in-sample and fail on val/test. The fix is
standard stacking practice: give the meta-model out-of-fold predictions, where every
train row's prediction comes from a model that never saw that row during fitting.

METHOD
------
5-fold StratifiedGroupKFold on train, grouped by subject_id (a patient's studies never
split across folds -- the same rule used for the original train/val/test split) and
stratified by final_label. For each fold, fit on the other 4 folds and predict the held-
out one; concatenating all 5 held-out predictions covers the full train set OOF.

Reuses the exact feature engineering and hyperparameters from the two baseline scripts
(imported, not re-implemented) so the OOF predictions are produced by the same models
already validated in Phase 2:
  - blood: HistGradientBoostingClassifier, scripts/train_blood_baseline.py
  - text:  LogisticRegression on frozen Bio_ClinicalBERT embeddings, Setting A
           (INDICATION section) ONLY -- scripts/train_text_baseline.py. The full_report
           setting is never used for fusion; it leaks label-derivation logic.

Run from repo root, after both baseline scripts have already been run once (text needs
its cached emb_indication_train.npy):  python scripts/generate_oof_predictions.py
"""
import sys
from pathlib import Path

# On Windows, sklearn's HistGradientBoostingClassifier (a Cython extension with its own
# bundled OpenMP runtime) fails torch's later DLL init (WinError 1114) if sklearn loads
# first. Importing torch before any sklearn import sidesteps it -- harmless elsewhere.
import torch  # noqa: F401,E402

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path("scripts").resolve()))
import train_blood_baseline as blood_mod   # noqa: E402
import train_text_baseline as text_mod     # noqa: E402

N_FOLDS = 5
SEED = 42
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]


def make_folds(df: pd.DataFrame) -> np.ndarray:
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1)
    for i, (_, heldout_idx) in enumerate(skf.split(df, df["y"], groups=df["subject_id"])):
        fold[heldout_idx] = i
    assert (fold >= 0).all()
    return fold


def oof_blood(train: pd.DataFrame, fold: np.ndarray) -> np.ndarray:
    X, y = train[blood_mod.FEATURES], train["y"].values
    oof = np.zeros((len(train), 3))
    for k in range(N_FOLDS):
        tr, ho = fold != k, fold == k
        clf = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
            class_weight="balanced", random_state=SEED,
        )
        clf.fit(X[tr], y[tr])
        oof[ho] = clf.predict_proba(X[ho])
        fold_f1 = f1_score(y[ho], clf.predict(X[ho]), average="macro")
        print(f"  [blood] fold {k}: n_heldout={ho.sum()}, macro F1={fold_f1:.4f}")
    return oof


def oof_text_indication(train: pd.DataFrame, fold: np.ndarray) -> np.ndarray:
    emb_path = text_mod.OUT_DIR / "emb_indication_train.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} missing -- run scripts/train_text_baseline.py once first "
            "so the indication embeddings are cached."
        )
    X = np.load(emb_path)
    y = train["y"].values
    assert len(X) == len(train), "cached embeddings don't match train.csv row count"
    oof = np.zeros((len(train), 3))
    for k in range(N_FOLDS):
        tr, ho = fold != k, fold == k
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED)
        clf.fit(X[tr], y[tr])
        oof[ho] = clf.predict_proba(X[ho])
        fold_f1 = f1_score(y[ho], clf.predict(X[ho]), average="macro")
        print(f"  [text/indication] fold {k}: n_heldout={ho.sum()}, macro F1={fold_f1:.4f}")
    return oof


def save(train: pd.DataFrame, fold: np.ndarray, proba: np.ndarray, prefix: str, out_path: Path) -> None:
    out = train[["subject_id", "study_id", "hadm_id", "final_label"]].copy()
    out["fold"] = fold
    for i, n in enumerate(LABEL_NAMES):
        out[f"{prefix}_p_{n.replace(' ', '_').lower()}"] = proba[:, i]
    out[f"{prefix}_pred"] = [LABEL_NAMES[i] for i in proba.argmax(axis=1)]
    out.to_csv(out_path, index=False)
    overall_f1 = f1_score(train["y"], proba.argmax(axis=1), average="macro")
    print(f"  saved {out_path}  (overall OOF macro F1: {overall_f1:.4f})")


def main() -> None:
    train = pd.read_csv("data/processed/train.csv")
    train["y"] = train["final_label"].map({"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2})
    train = blood_mod.add_derived(train)
    fold = make_folds(train)
    print(f"train n={len(train)}, {N_FOLDS} subject-grouped stratified folds")
    print("fold sizes:", pd.Series(fold).value_counts().sort_index().to_dict())

    print("\n=== blood OOF ===")
    blood_oof = oof_blood(train, fold)
    save(train, fold, blood_oof, "blood", blood_mod.OUT_DIR / "oof_train_predictions.csv")

    print("\n=== text (indication) OOF ===")
    text_oof = oof_text_indication(train, fold)
    save(train, fold, text_oof, "text_indication", text_mod.OUT_DIR / "oof_train_predictions.csv")

    print("\nDone. These, plus val_predictions.csv from each baseline (and the image "
          "model's OOF+val predictions, generated on Kaggle), feed scripts/train_fusion.py.")


if __name__ == "__main__":
    main()
