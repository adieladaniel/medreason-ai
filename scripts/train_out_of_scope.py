"""
Train and judge the out-of-scope detector. Plan, arms and adoption rule: experiments/out_of_scope/PLAN.md, written before this ran.

Training rows: the in-scope training studies (their out-of-fold predictions, label 0) and the Excluded studies of training patients (label 1).
The arm and the threshold are chosen by patient-grouped 5-fold cross-validation on those rows only. Then, once, the detector is judged on
the in-scope validation studies (negatives) against the Excluded studies of patients in no split (positives, never used for any choice), with
the Excluded studies of validation patients as a separate first check. The sealed test split is not opened (its patients cannot even be named).

The detector is written to disk, and its rules to app/data/out_of_scope_rules.json, only if it passes the adoption rule (`adopt`). If it does not,
nothing the app reads changes, and the result is reported as a negative one.

Run from the repo root, after scripts/extract_scope_features.py:   python scripts/train_out_of_scope.py
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import scope  # noqa: E402

OUT = ROOT / "experiments" / "out_of_scope"
SEED, B, FOLDS = 42, 2000, 5
FLAG_SHARE = 0.10                          # the operating point: flag this share of in-scope training studies
CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
SLUG = ["normal", "pneumonia", "lung_cancer"]
ARMS = {"A": {"with_backbone": False, "model": "logreg"}, "B": {"with_backbone": True, "model": "logreg"},
        "C": {"with_backbone": True, "model": "hgb"}}
TIE = 0.005


# ------------------------------------------------------------------ the rules written in PLAN.md
def adopt(m: dict) -> tuple[bool, list[str]]:
    """`m` holds, measured on in-scope validation against held-out Excluded studies: auroc, auroc_lo, catch_rate (share of the Excluded studies
    flagged), false_flag_rate (share of in-scope validation studies flagged), f1_unflagged, f1_all, cancers_flagged."""
    why = []
    if not (m["auroc"] >= 0.75 and m["auroc_lo"] >= 0.70):
        why.append("AUROC below 0.75, or its lower bound below 0.70")
    if not m["catch_rate"] >= 0.35:
        why.append("flags fewer than 35 percent of the held-out Excluded studies")
    if not m["false_flag_rate"] <= 0.13:
        why.append("flags more than 13 percent of in-scope validation studies")
    if not m["f1_unflagged"] >= m["f1_all"]:
        why.append("macro F1 on unflagged in-scope studies is below the macro F1 on all of them")
    if not m["cancers_flagged"] <= 0.20:
        why.append("flags more than 20 percent of true Lung Cancer validation studies")
    return (not why), why


def choose_arm(cv_auc: dict[str, float]) -> str:
    """Best cross-validated AUROC; within 0.005 of the best, the simpler arm (A before B before C)."""
    best = max(cv_auc.values())
    return next(a for a in ARMS if cv_auc[a] >= best - TIE)


# ------------------------------------------------------------------ data
def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=300, random_state=SEED)


def inscope(split: str) -> dict:
    """Features of the in-scope studies from the stored prediction files and the cached backbone outputs (training: out-of-fold)."""
    from explain.blood import RAW_LABS
    from explain.text import extract_indication
    table = pd.read_csv(ROOT / "data" / "processed" / f"{split}.csv")
    fname = "oof_train_predictions.csv" if split == "train" else "val_predictions.csv"
    parts = {}
    for m, prefix in (("image", "image_p_"), ("blood", "blood_p_"), ("text", "text_indication_p_")):
        d = pd.read_csv(ROOT / "experiments" / f"{m}_baseline" / fname).set_index("study_id").loc[table["study_id"]]
        parts[m] = d[[prefix + s for s in SLUG]].values.astype(np.float32)
    has_text = np.array([bool(re.search(r"\w", extract_indication(Path(p).read_text(encoding="utf-8", errors="ignore")) or ""))
                         for p in table["report_path"]])
    return {"study_id": table["study_id"].values, "subject_id": table["subject_id"].values,
            "xrv18": np.load(ROOT / "experiments" / "image_baseline" / f"feats_{split}.npy"), "image_p": parts["image"], "blood_p": parts["blood"],
            "text_p": parts["text"], "has_blood": table[RAW_LABS].notna().any(axis=1).values, "has_text": has_text,
            "y": table["final_label"].map({c: i for i, c in enumerate(CLASSES)}).values}


def excluded(role: str) -> dict:
    z = np.load(OUT / f"feats_{role}.npz")
    return {k: z[k] for k in z.files}


def matrix(d: dict, with_backbone: bool) -> np.ndarray:
    return scope.features(d["xrv18"], d["image_p"], d["blood_p"], d["text_p"], d["has_blood"], d["has_text"], with_backbone)


# ------------------------------------------------------------------ cross-validation on training rows only
def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold scores with whole patients held out together."""
    oof = np.zeros(len(y))
    for tr, ho in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y, groups):
        oof[ho] = make_model(kind).fit(X[tr], y[tr]).predict_proba(X[ho])[:, 1]
    return oof


# ------------------------------------------------------------------ judging
def judge(score_in: np.ndarray, score_ex: np.ndarray, thr: float, subj_in: np.ndarray, subj_ex: np.ndarray, in_pred: np.ndarray,
          in_true: np.ndarray, conf_in: np.ndarray, conf_ex: np.ndarray, b: int = B, seed: int = SEED) -> dict:
    """The quantities of the adoption rule with 95 percent intervals that resample patients, plus the existing confidence signal for context."""
    def stats(ii, ie):
        s_in, s_ex = score_in[ii], score_ex[ie]
        flagged = s_in >= thr
        keep = ~flagged
        cancers = in_true[ii] == 2
        return {"auroc": roc_auc_score(np.r_[np.zeros(len(s_in)), np.ones(len(s_ex))], np.r_[s_in, s_ex]),
                "catch_rate": float((s_ex >= thr).mean()), "false_flag_rate": float(flagged.mean()),
                "f1_all": float(f1_score(in_true[ii], in_pred[ii], average="macro", zero_division=0)),
                "f1_unflagged": float(f1_score(in_true[ii][keep], in_pred[ii][keep], average="macro", zero_division=0)),
                "cancers_flagged": float(flagged[cancers].mean()) if cancers.any() else float("nan"),
                "auroc_existing_confidence": roc_auc_score(np.r_[np.zeros(len(ii)), np.ones(len(ie))], np.r_[-conf_in[ii], -conf_ex[ie]])}
    gi = [np.where(subj_in == s)[0] for s in np.unique(subj_in)]
    ge = [np.where(subj_ex == s)[0] for s in np.unique(subj_ex)]
    point = stats(np.arange(len(score_in)), np.arange(len(score_ex)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(b):
        ii = np.concatenate([gi[k] for k in rng.integers(0, len(gi), len(gi))])
        ie = np.concatenate([ge[k] for k in rng.integers(0, len(ge), len(ge))])
        draws.append(stats(ii, ie))
    interval = {k: [float(np.nanpercentile([d[k] for d in draws], 2.5)), float(np.nanpercentile([d[k] for d in draws], 97.5))] for k in point}
    return {"point": point, "interval": interval}


def clean_diagnostic(va: dict, ex_va: dict, ex_none: dict) -> dict:
    """POST-HOC, added after the held-out result disagreed with cross-validation. The training rows mix two kinds of prediction: the in-scope
    negatives are out-of-fold, but the Excluded studies of training patients come from models that saw those patients, so a detector can learn
    "was this patient seen". This repeats the whole comparison on rows with no such mismatch: in-scope validation studies against Excluded
    studies of validation patients, every prediction made on patients the models never saw, cross-validated by patient. If the AUROC there is
    also low, the signal is not in these features and repairing the training rows would not help."""
    y = np.r_[np.zeros(len(va["study_id"])), np.ones(len(ex_va["study_id"]))]
    groups = np.r_[va["subject_id"], ex_va["subject_id"]]
    out = {"in_scope_validation": int((y == 0).sum()), "excluded_validation_patients": int((y == 1).sum()), "patients": int(len(np.unique(groups))),
           "cv_auroc": {}, "in_scope_oof_vs_no_split_excluded_auroc": {}}
    for arm, cfg in ARMS.items():
        X = np.vstack([matrix(va, cfg["with_backbone"]), matrix(ex_va, cfg["with_backbone"])])
        oof = cross_validate(X, y, groups, cfg["model"])
        s_none = make_model(cfg["model"]).fit(X, y).predict_proba(matrix(ex_none, cfg["with_backbone"]))[:, 1]
        out["cv_auroc"][arm] = float(roc_auc_score(y, oof))
        out["in_scope_oof_vs_no_split_excluded_auroc"][arm] = float(roc_auc_score(
            np.r_[np.zeros(int((y == 0).sum())), np.ones(len(s_none))], np.r_[oof[y == 0], s_none]))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tr, va = inscope("train"), inscope("val")
    ex_tr, ex_va, ex_none = excluded("train"), excluded("val"), excluded("none")
    y = np.r_[np.zeros(len(tr["study_id"])), np.ones(len(ex_tr["study_id"]))]
    groups = np.r_[tr["subject_id"], ex_tr["subject_id"]]
    print(f"training rows: {int((y == 0).sum())} in-scope, {int((y == 1).sum())} Excluded, {len(np.unique(groups))} patients")

    cv, oof = {}, {}
    for arm, cfg in ARMS.items():
        X = np.vstack([matrix(tr, cfg["with_backbone"]), matrix(ex_tr, cfg["with_backbone"])])
        oof[arm] = cross_validate(X, y, groups, cfg["model"])
        cv[arm] = float(roc_auc_score(y, oof[arm]))
        print(f"arm {arm}: cross-validated AUROC {cv[arm]:.3f}", flush=True)
    arm = choose_arm(cv)
    cfg = ARMS[arm]
    print(f"chosen arm: {arm}")

    X = np.vstack([matrix(tr, cfg["with_backbone"]), matrix(ex_tr, cfg["with_backbone"])])
    model = make_model(cfg["model"]).fit(X, y)
    thr = float(np.percentile(oof[arm][y == 0], 100 * (1 - FLAG_SHARE)))

    def score(d):
        return model.predict_proba(matrix(d, cfg["with_backbone"]))[:, 1]
    s_va, s_exva, s_none = score(va), score(ex_va), score(ex_none)
    pool_va = scope.pooled(va["image_p"], va["blood_p"], va["text_p"], va["has_blood"], va["has_text"])
    conf = lambda d: scope.pooled(d["image_p"], d["blood_p"], d["text_p"], d["has_blood"], d["has_text"]).max(axis=1)      # noqa: E731
    first_check = judge(s_va, s_exva, thr, va["subject_id"], ex_va["subject_id"], pool_va.argmax(1), va["y"], conf(va), conf(ex_va))
    final = judge(s_va, s_none, thr, va["subject_id"], ex_none["subject_id"], pool_va.argmax(1), va["y"], conf(va), conf(ex_none))

    m = {**final["point"], "auroc_lo": final["interval"]["auroc"][0]}
    ok, why = adopt(m)
    result = {"cv_auroc": cv, "chosen_arm": arm, "threshold": thr, "flag_share_in_scope_training": FLAG_SHARE, "training_rows": {
        "in_scope": int((y == 0).sum()), "excluded": int((y == 1).sum()), "patients": int(len(np.unique(groups)))},
        "first_check_validation_patients": first_check, "final_held_out_patients": final, "adopted": ok, "failed_because": why,
        "n": {"in_scope_validation": len(s_va), "excluded_validation_patients": len(s_exva), "excluded_held_out": len(s_none)}}
    result["post_hoc_clean_diagnostic"] = clean_diagnostic(va, ex_va, ex_none)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if ok:
        joblib.dump(model, OUT / "detector.joblib")
        rules = {"arm": arm, "with_backbone": cfg["with_backbone"], "model": cfg["model"], "threshold": thr, "feature_names": scope.feature_names(cfg["with_backbone"]),
                 "operating_point": {"flag_rate_in_scope": round(final["point"]["false_flag_rate"], 3),
                                     "catch_rate_held_out": round(final["point"]["catch_rate"], 3), "auroc_held_out": round(final["point"]["auroc"], 3)},
                 "built": "2026-10-29", "trained_on": "in-scope training out-of-fold predictions and Excluded studies of training patients"}
        (ROOT / "app" / "data" / "out_of_scope_rules.json").write_text(json.dumps(rules, indent=1), encoding="utf-8")

    p, i = final["point"], final["interval"]
    print(f"\nheld-out Excluded studies ({len(s_none)}) against in-scope validation ({len(s_va)}), threshold {thr:.3f}")
    for k in ("auroc", "auroc_existing_confidence", "catch_rate", "false_flag_rate", "f1_all", "f1_unflagged", "cancers_flagged"):
        print(f"  {k:28} {p[k]:.3f} ({i[k][0]:.3f} to {i[k][1]:.3f})")
    q = first_check["point"]
    print(f"first check (Excluded studies of validation patients, {len(s_exva)}): AUROC {q['auroc']:.3f}, catch rate {q['catch_rate']:.3f}")
    d = result["post_hoc_clean_diagnostic"]
    print("post-hoc clean diagnostic, cross-validated AUROC on rows with no seen-patient mismatch:", {a: round(v, 3) for a, v in d["cv_auroc"].items()})
    print("ADOPTED" if ok else "NOT ADOPTED: " + "; ".join(why))


if __name__ == "__main__":
    main()
