"""
The one evaluation of the whole system on the sealed test split. The plan, what is reported, and how the result will be read are in
experiments/test_evaluation/PLAN.md, written before this split was opened.

Two commands, in order:
    python scripts/eval_test_split.py --predict    # never loads the label column; resumable; writes predictions.csv
    python scripts/eval_test_split.py --score      # reads the labels once; writes results.json; refuses to run again

Why two: the predictions do not depend on the labels, so they can be stopped and resumed without anyone having seen a score. Once
the labels have been read and results.json exists, a second scoring would be a second look at the test split, and the script refuses.

The same `evaluate` function scores the validation split from its stored predictions, so validation and test are compared with
identical code. Two ways of forming the fused answer are reported, because the deployed app pools only the inputs a case has, while
the validation figure in the notes always pooled all three (see PLAN.md).
"""
import torch  # noqa: F401  (Windows: torch before sklearn and shap, see CLAUDE.md gotcha 7)

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "test_evaluation"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"      # a rehearsal on validation studies points this elsewhere
PREDICTIONS, RESULTS, FROZEN = OUT / "predictions.csv", OUT / "results.json", OUT / "frozen_system.json"
MODS = ("image", "blood", "text")
CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
SLUG = ["normal", "pneumonia", "lung_cancer"]
SEED, B = 42, 2000
ALERT_LEVELS = (0.5, 0.6, 0.7, 0.8)
FROZEN_FILES = ["experiments/blood_baseline/hgb_model.joblib", "experiments/text_baseline/logreg_indication.joblib",
                "experiments/image_baseline/image_head_final.pth", "app/data/decision_rules.json",
                "explain/image.py", "explain/blood.py", "explain/text.py", "app/agent/belief.py", "app/agent/reasoning.py",
                "app/agent/decision.py", "app/agent/consistency.py", "app/pipelines/staged.py"]
# The scoring code in this script is deliberately not frozen: a flaw found in it after scoring is fixed and shown both ways (PLAN.md).


# ------------------------------------------------------------------ what is frozen
BACKBONE = Path.home() / ".torchxrayvision" / "models_data" / "mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"


def frozen_system() -> dict:
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:                                                            # noqa: BLE001
        head = "unknown"
    return {"git_head": head, "files": {f: sha(ROOT / f) for f in FROZEN_FILES}, "backbone_weights": sha(BACKBONE)}


# ------------------------------------------------------------------ step 1: predictions, never reading a label
def load_inputs(path: Path) -> pd.DataFrame:
    """The test table without its label column. The label is not loaded at all in the prediction step."""
    return pd.read_csv(path, usecols=lambda c: c != "final_label")


def case_inputs(row: pd.Series) -> dict:
    from explain.blood import RAW_LABS
    from explain.text import extract_indication
    report = Path(row["report_path"]).read_text(encoding="utf-8", errors="ignore")
    return {"image": row["image_path"], "labs": {c: (None if pd.isna(row[c]) else float(row[c])) for c in RAW_LABS},
            "indication": extract_indication(report)}


def predict_row(row: pd.Series, predictors: dict) -> dict:
    """Each model's probabilities, always for all three (the validation convention needs them), and which inputs the case really has
    (the deployed app's rule: app.pipelines.staged.provided_inputs)."""
    from app.pipelines.staged import provided_inputs
    c = case_inputs(row)
    have = provided_inputs(c["indication"], c["labs"], c["image"])
    arg = {"image": c["image"], "blood": c["labs"], "text": c["indication"] or ""}
    rec = {"study_id": int(row["study_id"]), "subject_id": int(row["subject_id"]), "errors": ""}
    for m in MODS:
        try:
            p = predictors[m](arg[m])["probabilities"]
            for cls, s in zip(CLASSES, SLUG):
                rec[f"{m}_p_{s}"] = float(p[cls])
        except Exception as exc:                                                 # noqa: BLE001
            rec["errors"] += f"{m}: {type(exc).__name__}; "
            have.pop(m, None)
            for s in SLUG:
                rec[f"{m}_p_{s}"] = float("nan")
        rec[f"has_{m}"] = m in have
    return rec


def run_predict() -> None:
    from app.agent.tools import PREDICTORS
    OUT.mkdir(parents=True, exist_ok=True)
    if FROZEN.exists():
        before = json.loads(FROZEN.read_text(encoding="utf-8"))
        now = frozen_system()
        changed = [f for f in now["files"] if now["files"][f] != before["files"].get(f)] + \
                  (["backbone_weights"] if now["backbone_weights"] != before["backbone_weights"] else [])
        if changed:
            sys.exit(f"the system changed since the first prediction run: {changed}. Restore it, or start over by removing {FROZEN}.")
    else:
        FROZEN.write_text(json.dumps(frozen_system(), indent=1), encoding="utf-8")
    table = load_inputs(TEST_CSV)
    done = pd.read_csv(PREDICTIONS) if PREDICTIONS.exists() else pd.DataFrame()
    seen = set(done["study_id"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(table.iterrows(), 1):
        if int(row["study_id"]) in seen:
            continue
        rows.append(predict_row(row, PREDICTORS))
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(PREDICTIONS, index=False)
            print(f"  {len(rows)}/{len(table)} studies, {time.perf_counter() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(PREDICTIONS, index=False)
    print(f"predictions for {len(rows)} studies saved to {PREDICTIONS}; no label was read")


# ------------------------------------------------------------------ step 2: scoring
def pool(P: dict[str, np.ndarray], mask: dict[str, np.ndarray] | None, which=MODS) -> np.ndarray:
    """Normalised geometric mean over the chosen models (the app's rule). With `mask`, a model that was not given its input for a case
    is left out of that case's pool; without it, all chosen models count for every case."""
    num, den = 0.0, 0.0
    for m in which:
        w = np.ones(len(P[m])) if mask is None else mask[m].astype(float)
        num = num + w[:, None] * np.log(np.clip(P[m], 1e-6, 1.0))
        den = den + w
    g = np.exp(num / np.maximum(den, 1)[:, None])
    return g / g.sum(axis=1, keepdims=True)


def per_row(P: dict, mask: dict, convention: bool, rules: dict, cov: float) -> dict:
    """Everything about each case that does not depend on resampling: the fused probabilities, the decision layer's outputs and the
    consistency verdict. `convention` pools all three models for every case, as the validation figure did."""
    from app.agent.decision import apply_temperature, decide
    from app.agent.reasoning import reason
    p = pool(P, None if convention else mask)
    n = len(p)
    low, sets, cal = np.zeros(n, bool), np.zeros((n, 3), bool), np.zeros((n, 3))
    informative, flags = np.zeros(n, bool), {lv: np.zeros(n, bool) for lv in ALERT_LEVELS}
    status = np.empty(n, dtype=object)
    for i in range(n):
        have = list(MODS) if convention else [m for m in MODS if mask[m][i]]
        probs = dict(zip(CLASSES, p[i]))
        for lv in ALERT_LEVELS:
            d = decide(probs, have, rules=rules, set_coverage=cov, alert_recall=lv)
            if d is None:                                                # no rules for this combination of inputs (e.g. no input at all)
                if lv == 0.6:
                    low[i] = True
                continue
            flags[lv][i] = d["cancer_alert"]["flag"]
            if lv == 0.6:
                low[i], informative[i] = d["low_confidence"], d["cancer_alert"]["informative"]
                for c in d["prediction_set"]:
                    sets[i, CLASSES.index(c)] = True
                cal[i] = apply_temperature(p[i], rules["by_inputs"]["+".join(sorted(have))]["temperature"])
        status[i] = reason({m: dict(zip(CLASSES, P[m][i])) for m in have})["consistency"]["status"] if have else "no input"
    return {"p": p, "low": low, "sets": sets, "cal": cal, "informative": informative, "flags": flags, "status": status}


def macro_and_classes(y, pred):
    f = np.zeros(3)
    for c in range(3):
        tp = np.sum((pred == c) & (y == c))
        pr = tp / max(np.sum(pred == c), 1)
        rc = tp / max(np.sum(y == c), 1)
        f[c] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return f


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def brier(p, y):
    return float(((p - np.eye(3)[y]) ** 2).sum(1).mean())


def stats(idx: np.ndarray, y: np.ndarray, P: dict, rows: dict) -> dict:
    """All headline numbers on the cases `idx` (the whole set, or a bootstrap resample)."""
    yy, p, pred = y[idx], rows["p"][idx], rows["p"][idx].argmax(1)
    f = macro_and_classes(yy, pred)
    cancer = yy == 2
    out = {"macro_f1": float(f.mean()), "f1_normal": float(f[0]), "f1_pneumonia": float(f[1]), "f1_cancer": float(f[2]),
           "accuracy": float((pred == yy).mean()), "ece": ece(p, yy), "brier": brier(p, yy),
           "ece_calibrated": ece(rows["cal"][idx], yy), "brier_calibrated": brier(rows["cal"][idx], yy)}
    out["auc_cancer"] = float(roc_auc_score(cancer, p[:, 2])) if cancer.any() and (~cancer).any() else float("nan")
    for m in MODS:
        pm = P[m][idx]
        out[f"{m}_macro_f1"] = float(macro_and_classes(yy, pm.argmax(1)).mean())
        out[f"{m}_auc_cancer"] = float(roc_auc_score(cancer, pm[:, 2])) if cancer.any() and (~cancer).any() else float("nan")
    ans = ~rows["low"][idx]
    out["answered_share"] = float(ans.mean())
    out["accuracy_answered"] = float((pred[ans] == yy[ans]).mean()) if ans.any() else float("nan")
    out["macro_f1_answered"] = float(macro_and_classes(yy[ans], pred[ans]).mean()) if ans.any() else float("nan")
    out["cancers_declined"] = float(rows["low"][idx][cancer].mean()) if cancer.any() else float("nan")
    s = rows["sets"][idx]
    for c, name in enumerate(SLUG):
        out[f"set_coverage_{name}"] = float(s[yy == c, c].mean()) if (yy == c).any() else float("nan")
    out["single_class_set"] = float((s.sum(1) == 1).mean())
    out["mean_set_size"] = float(s.sum(1).mean())
    for lv in ALERT_LEVELS:
        fl = rows["flags"][lv][idx] & rows["informative"][idx]
        out[f"alert_{lv:g}_recall"] = float(fl[cancer].mean()) if cancer.any() else float("nan")
        out[f"alert_{lv:g}_per_100"] = float(fl.mean() * 100)
        out[f"alert_{lv:g}_share_real"] = float(cancer[fl].mean()) if fl.any() else float("nan")
    return out


def interval(x) -> list:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def evaluate(P: dict, mask: dict, y: np.ndarray, subjects: np.ndarray, rules: dict, cov: float = 0.8, b: int | None = None) -> dict:
    """Score one split, both ways of forming the fused answer. Pure: no files are read or written here."""
    b = B if b is None else b
    out = {"n": int(len(y)), "n_cancer": int((y == 2).sum()), "majority_class_accuracy": float(np.bincount(y, minlength=3).max() / len(y)),
           "inputs_present": {m: float(mask[m].mean()) for m in MODS}}
    uniq, inv = np.unique(subjects, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    draws_idx = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(b)]
    for name, convention in (("deployed", False), ("validation_convention", True)):
        rows = per_row(P, mask, convention, rules, cov)
        point = stats(np.arange(len(y)), y, P, rows)
        draws = [stats(ix, y, P, rows) for ix in draws_idx]
        res = {"point": point, "interval": {k: interval([d[k] for d in draws]) for k in point}}
        res["confusion"] = np.bincount(y * 3 + rows["p"].argmax(1), minlength=9).reshape(3, 3).tolist()
        res["consistency"] = {s: {"n": int((rows["status"] == s).sum()),
                                  "accuracy": float((rows["p"].argmax(1)[rows["status"] == s] == y[rows["status"] == s]).mean())}
                              for s in sorted(set(rows["status"]))}
        res["pairs_macro_f1"] = {"+".join(w): float(macro_and_classes(y, pool(P, None if convention else mask, w).argmax(1)).mean())
                                 for w in (("image", "blood"), ("image", "text"), ("blood", "text"), MODS)}
        out[name] = res
    return out


def validation_inputs() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Validation probabilities from the stored prediction files, and which inputs each case has, for the reference column."""
    from explain.blood import RAW_LABS
    from explain.text import extract_indication
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    cols = {"image": ("image", "image_p_"), "blood": ("blood", "blood_p_"), "text": ("text", "text_indication_p_")}
    P = {}
    for m, (d, prefix) in cols.items():
        s = pd.read_csv(ROOT / "experiments" / f"{d}_baseline" / "val_predictions.csv").set_index("study_id").loc[val["study_id"]]
        P[m] = s[[prefix + x for x in SLUG]].values
    has_text = np.array([bool(re.search(r"\w", extract_indication(Path(r).read_text(encoding="utf-8", errors="ignore")) or ""))
                         for r in val["report_path"]])
    mask = {"image": np.ones(len(val), bool), "blood": val[RAW_LABS].notna().any(axis=1).values, "text": has_text}
    return P, mask, val["final_label"].map({c: i for i, c in enumerate(CLASSES)}).values, val["subject_id"].values


def run_score() -> None:
    if RESULTS.exists():
        sys.exit(f"{RESULTS} exists: the test split has been scored, and a second scoring would be a second look at it. Not run.")
    if not PREDICTIONS.exists() or not FROZEN.exists():
        sys.exit("run --predict first")
    before, now = json.loads(FROZEN.read_text(encoding="utf-8")), frozen_system()
    if now["files"] != before["files"] or now["backbone_weights"] != before["backbone_weights"]:
        sys.exit("the system changed after the predictions were made; restore it before scoring")
    from app.agent.decision import load_rules
    rules = load_rules()
    pred = pd.read_csv(PREDICTIONS)
    labels = pd.read_csv(TEST_CSV, usecols=["study_id", "final_label"])         # the first read of the labels
    d = pred.merge(labels, on="study_id", how="inner")
    if len(d) != len(labels) or len(pred) != len(labels):
        sys.exit(f"predictions cover {len(pred)} of {len(labels)} studies; finish --predict first")
    y = d["final_label"].map({c: i for i, c in enumerate(CLASSES)}).values
    # a model that failed on a case has NaN probabilities and is marked absent; for the all-three convention a uniform vector is
    # neutral under the geometric-mean pool, so it neither helps nor hurts
    P = {m: np.where(np.isnan(d[[f"{m}_p_{s}" for s in SLUG]].values), 1 / 3, d[[f"{m}_p_{s}" for s in SLUG]].values) for m in MODS}
    mask = {m: d[f"has_{m}"].astype(bool).values for m in MODS}
    errors = int((d["errors"].fillna("") != "").sum())
    result = {"frozen_system": before, "prediction_errors": errors, "test": evaluate(P, mask, y, d["subject_id"].values, rules)}
    vP, vmask, vy, vsubjects = validation_inputs()
    result["validation_reference"] = evaluate(vP, vmask, vy, vsubjects, rules=rules)
    RESULTS.write_text(json.dumps(result, indent=1), encoding="utf-8")
    report(result)


def report(result: dict) -> None:
    print(f"prediction errors: {result['prediction_errors']}")
    for name in ("deployed", "validation_convention"):
        print(f"\n== {name}: validation against test")
        for key in ("macro_f1", "f1_normal", "f1_pneumonia", "f1_cancer", "accuracy", "auc_cancer", "ece", "ece_calibrated", "answered_share",
                    "accuracy_answered", "set_coverage_lung_cancer", "single_class_set", "alert_0.6_recall", "alert_0.6_per_100", "alert_0.6_share_real"):
            t, v = result["test"][name], result["validation_reference"][name]
            print(f"  {key:26} validation {v['point'][key]:.3f}   test {t['point'][key]:.3f} ({t['interval'][key][0]:.3f} to {t['interval'][key][1]:.3f})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--predict", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    run_predict() if a.predict else run_score()


if __name__ == "__main__":
    main()
