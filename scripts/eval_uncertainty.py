"""
Phase 8 groundwork, run on the VALIDATION set only (the test split stays sealed): how sure should
we be about the fusion's numbers, and when should the system decline to answer?

Point estimates such as "macro F1 0.613" hide how much they would move on a different sample,
which matters most for the rare Lung Cancer class (about 100 val cases). This script adds:

1. Confidence intervals. Cluster bootstrap that resamples SUBJECTS, not studies, because one
   patient contributes several studies and a plain bootstrap would give intervals that are too
   narrow. Reported for the fusion and each single modality.
2. Calibration. Brier score, log loss and expected calibration error of the pooled
   probabilities, then temperature scaling fitted on the out-of-fold TRAIN predictions only and
   applied to val. Scaling never changes the predicted class, so F1 is untouched.
3. Selective prediction. Abstain when confidence is below a threshold chosen on train to keep
   80% coverage, and report what that does on val: accuracy and macro F1 on the answered
   cases, and what share of each class is declined (a cancer-blind abstention rule would be bad).
4. Conformal prediction sets. Class-conditional (Mondrian) split conformal, calibrated on the
   out-of-fold train predictions. A set with one class is a confident answer, a set with two or
   three says "not sure between these". Class-conditional so the rare class gets its own
   coverage guarantee, which a single pooled threshold would not give it.
5. Subgroups. Fusion accuracy and macro F1 by sex, age band and recorded race group, with the
   number of cases and Lung Cancer cases so thin groups are visible.

Honest limits. The out-of-fold train predictions come from fold models, the val predictions from
the full-train models, so calibration on train transfers only approximately; the val coverage
numbers say how well it did. Conformal guarantees assume exchangeability and the patients repeat
across studies, so coverage is checked, not assumed. Subgroup differences with few cases are
noise: intervals are shown, and no claim of fairness or unfairness is made. Race is the recorded
administrative field in MIMIC-IV, used only to check performance gaps.

Run from repo root:  python scripts/eval_uncertainty.py
"""
import torch  # noqa: F401  (Windows: torch before sklearn, see CLAUDE.md gotcha 7)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from scripts.train_fusion import CLS, LABEL_NAMES, SOURCES, assemble  # noqa: E402

OUT = Path("experiments/uncertainty")
B = 2000
SEED = 42
BINS = 10
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ data
def pooled(X: np.ndarray, modalities=(0, 1, 2)) -> np.ndarray:
    """Normalised geometric mean over the chosen modalities: the belief state's rule."""
    s = X.reshape(len(X), 3, 3)[:, list(modalities), :].clip(1e-6)
    g = np.exp(np.log(s).mean(axis=1))
    return g / g.sum(axis=1, keepdims=True)


def demographics(df: pd.DataFrame) -> pd.DataFrame:
    pat = pd.read_csv("data/raw/mimic-iv/patients.csv", usecols=["subject_id", "gender", "anchor_age"])
    adm = pd.read_csv("data/raw/mimic-iv/admissions.csv", usecols=["hadm_id", "race"])
    d = df[["study_id", "subject_id", "hadm_id"]].merge(pat, on="subject_id", how="left").merge(
        adm.drop_duplicates("hadm_id"), on="hadm_id", how="left")
    d["age_band"] = pd.cut(d["anchor_age"], [0, 49, 64, 79, 200], labels=["under 50", "50 to 64", "65 to 79", "80 and over"])
    r = d["race"].fillna("UNKNOWN").str.upper()
    d["race_group"] = np.select(
        [r.str.contains("WHITE"), r.str.contains("BLACK"), r.str.contains("HISPANIC|LATINO"), r.str.contains("ASIAN")],
        ["White", "Black", "Hispanic or Latino", "Asian"], default="Other or unknown")
    return d


# ------------------------------------------------------------------ metrics
def confusion(y, pred):
    return np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)


def f1s(y, pred):
    """Per-class F1. A class with no true cases in the sample is NaN, not 0, so a small subgroup
    that happens to contain no cancer is not punished for it; macro F1 is the mean over the rest."""
    c = confusion(y, pred)
    tp = np.diag(c).astype(float)
    p = np.divide(tp, c.sum(0), out=np.zeros(3), where=c.sum(0) > 0)
    r = np.divide(tp, c.sum(1), out=np.zeros(3), where=c.sum(1) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(3), where=(p + r) > 0)
    f[c.sum(1) == 0] = np.nan
    return f


def recalls(y, pred):
    c = confusion(y, pred)
    return np.divide(np.diag(c), c.sum(1), out=np.full(3, np.nan), where=c.sum(1) > 0)


def ece(p, y, bins=BINS):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return tot


def brier(p, y):
    return float(((p - np.eye(3)[y]) ** 2).sum(1).mean())


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y].clip(1e-9)).mean())


def stat_all(p, y):
    pred = p.argmax(1)
    f = f1s(y, pred)
    out = {"macro_f1": np.nanmean(f), "f1_normal": f[0], "f1_pneumonia": f[1], "f1_cancer": f[2],
           "accuracy": (pred == y).mean(), "brier": brier(p, y), "ece": ece(p, y)}
    if (y == 2).any() and (y != 2).any():
        out["auc_cancer"] = roc_auc_score(y == 2, p[:, 2])
    return out


def cluster_bootstrap(fn, subjects, b=B):
    """fn(index_array) -> dict of numbers. Resamples whole subjects with replacement."""
    uniq, inv = np.unique(subjects, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    draws = []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        draws.append(fn(np.concatenate([groups[i] for i in pick])))
    keys = draws[0].keys()
    return {k: (float(np.nanpercentile([d[k] for d in draws], 2.5)), float(np.nanpercentile([d[k] for d in draws], 97.5)))
            for k in keys}


def fmt(v, ci):
    return f"{v:.3f} ({ci[0]:.3f} to {ci[1]:.3f})"


# ------------------------------------------------------------------ calibration
def fit_temperature(p_train, y_train):
    """One scalar T minimising log loss of softmax(log p / T) on the out-of-fold train predictions."""
    lp = np.log(p_train.clip(1e-9))
    best, best_t = np.inf, 1.0
    for t in np.linspace(0.5, 4.0, 141):
        z = lp / t
        z = np.exp(z - z.max(1, keepdims=True))
        q = z / z.sum(1, keepdims=True)
        v = nll(q, y_train)
        if v < best:
            best, best_t = v, t
    return float(best_t)


def apply_temperature(p, t):
    z = np.log(p.clip(1e-9)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


# ------------------------------------------------------------------ conformal
def conformal_quantiles(P, y, alpha):
    """Per-class threshold on the score 1 - p(class): the finite-sample-corrected (1 - alpha) quantile
    among calibration cases whose TRUE class is that class (class-conditional, Mondrian)."""
    out = []
    for c in range(3):
        n = int((y == c).sum())
        out.append(np.quantile(1 - P[y == c, c], min(1.0, (1 - alpha) * (1 + 1 / n))))
    return np.array(out)


def prediction_sets(P, q):
    """Boolean [n, 3]: class c is in the set when its score is within class c's threshold."""
    return (1 - P) <= q[None, :]


# ------------------------------------------------------------------ main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, Xtr, ytr = assemble("train")
    va, Xva, yva = assemble("val")
    # subject ids for the cluster bootstrap and the demographics
    ids = pd.read_csv("data/processed/val.csv", usecols=["study_id", "subject_id", "hadm_id"])
    va = va.merge(ids, on="study_id", how="left")
    assert va["subject_id"].notna().all()
    demo = demographics(va)
    P = pooled(Xva)
    ref = pd.read_csv("experiments/fusion/val_predictions.csv").set_index("study_id").loc[va["study_id"], "fused_pred"].values
    assert (P.argmax(1) == pd.Series(ref).map({"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}).values).all() or \
        (P.argmax(1) == np.asarray(ref)).all(), "pooled prediction must match the saved fusion"
    base = stat_all(P, yva)
    print(f"val n={len(yva)} subjects={va['subject_id'].nunique()} fusion macro F1 {base['macro_f1']:.4f} (saved: 0.6133)")
    assert abs(base["macro_f1"] - 0.6133) < 0.002

    subj = va["subject_id"].values
    res = {"n": int(len(yva)), "subjects": int(va["subject_id"].nunique()), "bootstrap_draws": B}

    # ---- 1. confidence intervals: fusion and each modality
    ci = {}
    systems = {"fusion": P, **{m: pooled(Xva, (i,)) for i, m in enumerate(SOURCES)}}
    for name, p in systems.items():
        pt = stat_all(p, yva)
        bs = cluster_bootstrap(lambda ix, p=p: stat_all(p[ix], yva[ix]), subj)
        ci[name] = {k: {"value": float(pt[k]), "lo": bs[k][0], "hi": bs[k][1]} for k in pt}
        print(f"{name:<7} macro F1 {fmt(pt['macro_f1'], bs['macro_f1'])}  cancer F1 {fmt(pt['f1_cancer'], bs['f1_cancer'])}  "
              f"cancer AUC {fmt(pt['auc_cancer'], bs['auc_cancer'])}")
    # paired difference: is fusion really above the best single model?
    best_single = max(SOURCES, key=lambda m: ci[m]["macro_f1"]["value"])
    diff = cluster_bootstrap(lambda ix: {"d": stat_all(P[ix], yva[ix])["macro_f1"] - stat_all(systems[best_single][ix], yva[ix])["macro_f1"]}, subj)
    res["fusion_minus_best_single"] = {"best_single": best_single,
        "value": ci["fusion"]["macro_f1"]["value"] - ci[best_single]["macro_f1"]["value"], "lo": diff["d"][0], "hi": diff["d"][1]}
    print(f"fusion minus {best_single}: {res['fusion_minus_best_single']['value']:+.3f} (95% CI {diff['d'][0]:+.3f} to {diff['d'][1]:+.3f})")
    res["ci"] = ci

    # ---- 2. calibration
    Ptr = pooled(Xtr)
    T = fit_temperature(Ptr, ytr)
    Pc = apply_temperature(P, T)
    cal = {"temperature": T}
    for nm, p in (("raw", P), ("temperature_scaled", Pc)):
        pt = stat_all(p, yva)
        bs = cluster_bootstrap(lambda ix, p=p: {"ece": ece(p[ix], yva[ix]), "brier": brier(p[ix], yva[ix]), "nll": nll(p[ix], yva[ix])}, subj, 1000)
        cal[nm] = {"ece": [float(pt["ece"]), *bs["ece"]], "brier": [float(pt["brier"]), *bs["brier"]],
                   "nll": [nll(p, yva), *bs["nll"]]}
        print(f"calibration {nm:<18} ECE {fmt(pt['ece'], bs['ece'])}  Brier {pt['brier']:.3f}  NLL {nll(p, yva):.3f}")
    print(f"temperature T = {T:.2f} ({'T < 1 sharpens: the pooled probabilities were under-confident' if T < 1 else 'T > 1 softens: they were over-confident'})")
    res["calibration"] = cal

    # ---- 3. selective prediction
    conf_tr, conf = Pc.max(1) if False else apply_temperature(Ptr, T).max(1), Pc.max(1)
    pred = Pc.argmax(1)
    curve = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        tau = np.quantile(conf, 1 - cov) if cov < 1 else 0.0
        keep = conf >= tau
        f = f1s(yva[keep], pred[keep])
        curve.append({"coverage": float(keep.mean()), "accuracy": float((pred[keep] == yva[keep]).mean()),
                      "macro_f1": float(np.nanmean(f)), "f1_cancer": float(f[2]),
                      "kept_cancer": float(keep[yva == 2].mean())})
    res["risk_coverage"] = curve
    tau80 = float(np.quantile(conf_tr, 0.2))          # chosen on train, applied to val
    keep = conf >= tau80
    f = f1s(yva[keep], pred[keep])
    bs = cluster_bootstrap(lambda ix: {"cov": (conf[ix] >= tau80).mean(),
                                       "acc": (pred[ix][conf[ix] >= tau80] == yva[ix][conf[ix] >= tau80]).mean(),
                                       "kept_cancer": (conf[ix][yva[ix] == 2] >= tau80).mean()}, subj, 1000)
    res["abstain_rule"] = {"threshold": tau80, "target_coverage": 0.8, "val_coverage": float(keep.mean()),
                           "val_coverage_ci": list(bs["cov"]), "accuracy_answered": float((pred[keep] == yva[keep]).mean()),
                           "accuracy_all": float((pred == yva).mean()), "macro_f1_answered": float(np.nanmean(f)),
                           "macro_f1_all": base["macro_f1"],
                           "declined_by_class": {LABEL_NAMES[c]: float(1 - keep[yva == c].mean()) for c in range(3)},
                           "kept_cancer_ci": list(bs["kept_cancer"])}
    a = res["abstain_rule"]
    print(f"abstain below {tau80:.2f} (80% coverage on train): val coverage {a['val_coverage']:.2f}, accuracy "
          f"{a['accuracy_all']:.3f} -> {a['accuracy_answered']:.3f}, macro F1 {a['macro_f1_all']:.3f} -> {a['macro_f1_answered']:.3f}, "
          f"declined by class {{{', '.join(f'{k} {v:.0%}' for k, v in a['declined_by_class'].items())}}}")

    # ---- 4. class-conditional conformal sets, calibrated on out-of-fold train
    conformal = {}
    for alpha in (0.2, 0.1):
        q = conformal_quantiles(Ptr, ytr, alpha)
        sets = prediction_sets(P, q)
        size = sets.sum(1)
        covered = sets[np.arange(len(yva)), yva]
        bs = cluster_bootstrap(lambda ix: {f"cov_{LABEL_NAMES[c]}": sets[ix][yva[ix] == c, c].mean() for c in range(3)}, subj, 1000)
        conformal[str(alpha)] = {
            "target": 1 - alpha, "quantiles": q.round(4).tolist(),
            "coverage": {LABEL_NAMES[c]: {"value": float(covered[yva == c].mean()), "lo": bs[f"cov_{LABEL_NAMES[c]}"][0],
                                          "hi": bs[f"cov_{LABEL_NAMES[c]}"][1]} for c in range(3)},
            "mean_set_size": float(size.mean()),
            "share_empty": float((size == 0).mean()), "share_single": float((size == 1).mean()),
            "share_two": float((size == 2).mean()), "share_three": float((size == 3).mean()),
            "accuracy_when_single": float((sets[size == 1].argmax(1) == yva[size == 1]).mean()),
            "cancer_in_set_when_true": float(sets[yva == 2, 2].mean())}
        c = conformal[str(alpha)]
        print(f"conformal target {1 - alpha:.0%}: coverage " + ", ".join(f"{k} {v['value']:.2f}" for k, v in c["coverage"].items()) +
              f" | set size {c['mean_set_size']:.2f} (single {c['share_single']:.0%}, two {c['share_two']:.0%}, three {c['share_three']:.0%}) "
              f"| accuracy when single {c['accuracy_when_single']:.2f}")
    res["conformal"] = conformal

    # ---- 5. subgroups
    groups = []
    for col, label in (("gender", "Sex"), ("age_band", "Age"), ("race_group", "Recorded race")):
        for g in demo[col].dropna().unique():
            ix = np.where(demo[col].values == g)[0]
            if len(ix) < 30:
                continue
            pred_g = P[ix].argmax(1)
            bs = cluster_bootstrap(lambda j, ix=ix: {"macro_f1": np.nanmean(f1s(yva[ix][j], P[ix][j].argmax(1))),
                                                     "accuracy": (P[ix][j].argmax(1) == yva[ix][j]).mean()}, subj[ix], 1000)
            rec = recalls(yva[ix], pred_g)
            n_c = int((yva[ix] == 2).sum())
            groups.append({"attribute": label, "group": str(g), "n": int(len(ix)), "cancer_cases": n_c,
                           "thin": n_c < 10,           # too few Lung Cancer cases to read the group's macro F1
                           "macro_f1": float(np.nanmean(f1s(yva[ix], pred_g))), "macro_f1_ci": list(bs["macro_f1"]),
                           "accuracy": float((pred_g == yva[ix]).mean()), "accuracy_ci": list(bs["accuracy"]),
                           "recall": {LABEL_NAMES[c]: (None if np.isnan(rec[c]) else float(rec[c])) for c in range(3)}})
            r = groups[-1]
            print(f"{label:<14} {r['group']:<20} n={r['n']:>4} cancer={r['cancer_cases']:>3}{' (thin)' if r['thin'] else '       '} "
                  f"macro F1 {fmt(r['macro_f1'], r['macro_f1_ci'])}  acc {fmt(r['accuracy'], r['accuracy_ci'])}  "
                  f"recall N/P/C " + "/".join('n/a' if v is None else f'{v:.2f}' for v in r['recall'].values()))
    res["subgroups"] = groups

    (OUT / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
