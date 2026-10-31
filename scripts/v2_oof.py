"""
Step 2 of the version 2 plan (experiments/v2_four_class/PLAN.md): out-of-fold predictions for both arms, and the choice between them on training
data only.

  J   each modality retrained on four classes (the Excluded studies are the fourth), with the version 1 recipe.
  TB  one extra binary model per modality, Other versus in-scope, same recipe.
  T3  the three-class models on in-scope studies only. Out-of-fold on the new folds (the version 1 comparison), and for the validation and held-out
      studies the SAVED version 1 models, so arm T leaves the audited system untouched by construction.

For each: out-of-fold probabilities for every training row (a patient's studies are never in the fold that predicts them), and probabilities for the
validation and held-out rows from a model trained on all training rows. The validation and held-out probabilities are saved here and not scored: the
only scoring of those rows is scripts/v2_evaluate.py --score, once. The arm is chosen on the out-of-fold AUROC of the fused "other" score alone, and
a tie within 0.005 goes to T (PLAN.md). Resumable: a finished model's file is not recomputed.

Run from the repo root, after scripts/v2_embed_excluded.py:  python scripts/v2_oof.py
Output: experiments/v2_four_class/preds/<arm>_<modality>.npz (git-ignored), arm_choice.json, oof_summary.json.
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_lib as lib  # noqa: E402

TIE = 0.005
STORED_VAL = {"blood": "experiments/blood_baseline/val_predictions.csv", "text": "experiments/text_baseline/val_predictions.csv",
              "image": "experiments/image_baseline/val_predictions.csv"}


def fit_eval(modality, X_tr, y_tr, n_classes, X_eval):
    """-> (list of mean probabilities, per-seed probabilities for the image else None)."""
    if modality != "image":
        return lib.fit_predict(modality, X_tr, y_tr, n_classes, X_eval), None
    per_seed = lib.fit_predict_image_seeds(X_tr, y_tr, n_classes, X_eval)
    return [np.mean([s[i] for s in per_seed], axis=0) for i in range(len(X_eval))], per_seed


def run_arm(arm, modality, X, y, fold):
    """OOF on the training rows plus final-model predictions for validation and held-out, for one arm and modality."""
    n_classes = {"J": 4, "TB": 2, "T3": 3}[arm]
    target = (y == 3).astype(int) if arm == "TB" else y
    usable = (y < 3) if arm == "T3" else np.ones(len(y), bool)               # T3 learns from in-scope studies only
    Xtr = X["train"][modality]
    oof = np.zeros((len(y), n_classes))
    seeds = np.zeros((len(lib.IMAGE_SEEDS), len(y), n_classes)) if modality == "image" else None
    for k in range(lib.N_FOLDS):
        tr, ho = (fold != k) & usable, fold == k
        mean, per = fit_eval(modality, Xtr[tr], target[tr], n_classes, [Xtr[ho]])
        oof[ho] = mean[0]
        if per is not None:
            for s, p in enumerate(per):
                seeds[s, ho] = p[0]
    result = {"oof": oof}
    if seeds is not None:
        result["oof_seeds"] = seeds
    if arm == "T3":
        result.update(saved_version_1(modality, X))
    else:
        (val, held), _ = fit_eval(modality, Xtr[usable], target[usable], n_classes, [X["val"][modality], X["heldout"][modality]])
        result.update(val=val, heldout=held)
    return result


def saved_version_1(modality, X) -> dict:
    """Probabilities from the saved version 1 model for the validation and held-out rows, and how well they reproduce the stored version 1
    validation predictions (so a mismatch between this code's inputs and the audited system would show)."""
    root = lib.ROOT
    out = {}
    for role in ("val", "heldout"):
        x = X[role][modality]
        if modality == "blood":
            p = joblib.load(root / "experiments/blood_baseline/hgb_model.joblib").predict_proba(x)
        elif modality == "text":
            p = joblib.load(root / "experiments/text_baseline/logreg_indication.joblib").predict_proba(x)
        else:
            head = lib._head(3)
            head.load_state_dict(torch.load(root / "experiments/image_baseline/image_head_final.pth", map_location="cpu", weights_only=True))
            p = lib.head_proba(head.eval(), x)
        out[role] = p
    return out


def check_against_stored(modality, tables, val_probs) -> float:
    stored = pd.read_csv(lib.ROOT / STORED_VAL[modality])
    cols = [c for c in stored.columns if c.endswith(("_normal", "_pneumonia", "_lung_cancer")) and "_p_" in c and "full_report" not in c]
    assert len(cols) == 3, cols
    v = tables["val"]
    n_in = int((v.source == "in_scope").sum())
    s = stored.set_index("study_id").loc[v.study_id.values[:n_in], cols].values
    return float(np.abs(s - val_probs[:n_in]).max())


def choose(auroc_j: float, auroc_t: float) -> str:
    """J only if it is better by more than the tie margin; otherwise T, which leaves the audited system untouched."""
    return "J" if auroc_j - auroc_t > TIE else "T"


def other_score(arm, P):
    """P: dict modality -> probabilities of that arm. The fused probability of 'other' for each row."""
    stack = np.stack([P[m] for m in lib.MODALITIES], axis=1)
    pooled = lib.logpool(stack)
    return pooled[:, 3] if arm == "J" else pooled[:, 1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lib.PREDS.mkdir(parents=True, exist_ok=True)
    tables = lib.load_tables()
    X = lib.load_inputs(tables)
    y, fold = lib.labels(tables["train"]), tables["train"].fold.values
    summary = {"seconds": {}, "agreement_with_stored_version_1_validation": {}}
    for arm in ("J", "TB", "T3"):
        for m in lib.MODALITIES:
            f = lib.PREDS / f"{arm}_{m}.npz"
            if f.exists():
                continue
            t0 = time.time()
            res = run_arm(arm, m, X, y, fold)
            np.savez(f, **res)
            summary["seconds"][f"{arm}_{m}"] = round(time.time() - t0, 1)
            print(f"{arm} {m}: {summary['seconds'][f'{arm}_{m}']} s", flush=True)

    load = lambda arm, m: np.load(lib.PREDS / f"{arm}_{m}.npz")                  # noqa: E731
    for m in lib.MODALITIES:
        summary["agreement_with_stored_version_1_validation"][m] = round(check_against_stored(m, tables, load("T3", m)["val"]), 8)

    is_other = (y == 3).astype(int)
    choice = {"criterion": "AUROC of the fused 'other' score on the training out-of-fold predictions; a tie within 0.005 goes to T"}
    for arm in ("J", "T"):
        src = {"J": "J", "T": "TB"}[arm]
        P = {m: load(src, m)["oof"] for m in lib.MODALITIES}
        choice[f"auroc_{arm}"] = round(float(roc_auc_score(is_other, other_score(arm if arm == "J" else "TB", P))), 4)
        choice[f"auroc_{arm}_by_modality"] = {m: round(float(roc_auc_score(is_other, P[m][:, 3 if arm == "J" else 1])), 4)
                                                for m in lib.MODALITIES}
    choice["chosen"] = choose(choice["auroc_J"], choice["auroc_T"])
    (lib.OUT / "arm_choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print(json.dumps(choice, indent=1))

    # Information only: three-class macro F1 on the in-scope training rows, out-of-fold, for each modality, the recomputed version 1 and arm J
    # (its first three probabilities renormalised), and the seed spread of the image head.
    ins = y < 3
    info = {}
    for m in lib.MODALITIES:
        v1 = load("T3", m)["oof"][ins].argmax(1)
        j = load("J", m)["oof"][ins][:, :3].argmax(1)
        info[m] = {"macro_f1_version_1_recomputed": round(float(f1_score(y[ins], v1, average="macro")), 4),
                   "macro_f1_arm_J_first_three": round(float(f1_score(y[ins], j, average="macro")), 4)}
    seeds = load("J", "image")["oof_seeds"]
    info["image_seed_spread_auroc_other"] = [round(float(roc_auc_score(is_other, s[:, 3])), 4) for s in seeds]
    info["image_seed_spread_macro_f1_in_scope"] = [round(float(f1_score(y[ins], s[ins][:, :3].argmax(1), average="macro")), 4) for s in seeds]
    summary["oof_information"] = info
    (lib.OUT / "oof_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
