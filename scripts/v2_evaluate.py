"""
Steps 3 and 4 of the version 2 plan (experiments/v2_four_class/PLAN.md): fuse the chosen arm's predictions, set the operating point on training
data, and judge the adoption rule on the validation studies and the held-out patients.

  --rehearse   the whole machinery on the training out-of-fold predictions of fold 0 (its in-scope studies as negatives, its Excluded studies as
               positives, the threshold and temperature fitted on folds 1 to 4). Touches no validation or held-out row. Prints, writes nothing.
  --freeze     records the hashes of the chosen arm and of every prediction file in frozen.json. Must happen before --score.
  --score      the one real scoring of the validation and held-out rows. Refuses unless frozen.json matches the files, and refuses to run twice
               (results.json exists). Do not delete results.json to look again: a second look is no longer a test.

The fused "other" score is the pooled fourth probability for arm J and the pooled binary score for arm T. The in-scope class is the best of the
three in-scope classes, so a flag never changes it. The scoring code itself is not frozen, so a flaw found afterwards can be fixed and shown both ways.
"""
import torch  # noqa: F401  (Windows: torch before sklearn, see CLAUDE.md gotcha 7)

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_lib as lib  # noqa: E402

OUT = lib.OUT
B, SEED = 2000, 42
FLAG_RATE = 0.10                                     # the operating point: flag this share of in-scope training studies
OTHER_RATES = (0.05, 0.20)                           # reported for information
LIMITS = {"auroc": 0.75, "auroc_lower": 0.70, "flagged_heldout": 0.35, "flagged_inscope": 0.13, "macro_f1": 0.583,
          "cancer_flagged": 0.20, "ece": 0.05}


# ------------------------------------------------------------------ fusion
def load_arm(arm: str, key: str) -> dict:
    """modality -> array of that arm's probabilities for the 'oof', 'val' or 'heldout' rows."""
    src = {"J": "J", "T": "TB"}[arm]
    return {m: np.load(lib.PREDS / f"{src}_{m}.npz")[key] for m in lib.MODALITIES}


def load_three_class(key: str) -> dict:
    return {m: np.load(lib.PREDS / f"T3_{m}.npz")[key] for m in lib.MODALITIES}


def fuse(arm: str, key: str) -> dict:
    """The fused 'other' score and the in-scope class probabilities for one set of rows."""
    main = lib.logpool(np.stack([load_arm(arm, key)[m] for m in lib.MODALITIES], axis=1))
    v1 = lib.logpool(np.stack([load_three_class(key)[m] for m in lib.MODALITIES], axis=1))
    if arm == "J":
        p3 = main[:, :3] / main[:, :3].sum(1, keepdims=True)
        return {"other": main[:, 3], "p3": p3, "v1_p3": v1}
    return {"other": main[:, 1], "p3": v1, "v1_p3": v1}


def threshold_for(scores: np.ndarray, rate: float) -> float:
    """Flag when score >= tau. tau flags `rate` of these scores."""
    return float(np.quantile(scores, 1.0 - rate))


# ------------------------------------------------------------------ statistics
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def temperature(p: np.ndarray, y: np.ndarray) -> float:
    best, best_t = np.inf, 1.0
    for t in np.linspace(0.5, 4.0, 141):
        q = apply_temperature(p, t)
        v = float(-np.log(q[np.arange(len(y)), y].clip(1e-12)).mean())
        if v < best:
            best, best_t = v, t
    return float(best_t)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(p.clip(1e-9)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def groups_of(subjects: np.ndarray) -> list:
    uniq, inv = np.unique(subjects, return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(uniq))]


def draw(groups: list, rng) -> np.ndarray:
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def statistics(neg: dict, pos: dict, tau: float, t: float, iv=None, ih=None) -> dict:
    """The adoption statistics for the given rows (all rows when no index is given). neg: in-scope rows with 'other', 'p3', 'y' (0 to 2);
    pos: Excluded rows with 'other'."""
    iv = np.arange(len(neg["y"])) if iv is None else iv
    ih = np.arange(len(pos["other"])) if ih is None else ih
    y, s_neg, s_pos = neg["y"][iv], neg["other"][iv], pos["other"][ih]
    auroc = roc_auc_score(np.r_[np.zeros(len(s_neg)), np.ones(len(s_pos))], np.r_[s_neg, s_pos])
    cancers = y == 2
    return {"auroc": float(auroc), "flagged_heldout": float((s_pos >= tau).mean()), "flagged_inscope": float((s_neg >= tau).mean()),
            "macro_f1": float(f1_score(y, neg["p3"][iv].argmax(1), average="macro")),
            "cancer_flagged": float((s_neg[cancers] >= tau).mean()) if cancers.any() else float("nan"),
            "ece": ece(apply_temperature(neg["p3"][iv], t), y)}


def with_intervals(neg: dict, pos: dict, tau: float, t: float, neg_subjects, pos_subjects, b=B, seed=SEED) -> dict:
    """Point estimates and 95 percent intervals from a bootstrap that resamples patients (separately in the two sets)."""
    point = statistics(neg, pos, tau, t)
    rng = np.random.default_rng(seed)
    gn, gp = groups_of(neg_subjects), groups_of(pos_subjects)
    draws = [statistics(neg, pos, tau, t, draw(gn, rng), draw(gp, rng)) for _ in range(b)]
    out = {}
    for k, v in point.items():
        vals = [d[k] for d in draws]
        out[k] = {"value": v, "low": float(np.nanpercentile(vals, 2.5)), "high": float(np.nanpercentile(vals, 97.5))}
    return out


def adopt(result: dict) -> tuple[bool, dict]:
    """The pre-registered adoption rule. result maps each statistic to {'value', 'low', 'high'}."""
    r = {k: v["value"] for k, v in result.items()}
    checks = {
        "auroc": bool(r["auroc"] >= LIMITS["auroc"] and result["auroc"]["low"] >= LIMITS["auroc_lower"]),
        "flag_rates": bool(r["flagged_heldout"] >= LIMITS["flagged_heldout"] and r["flagged_inscope"] <= LIMITS["flagged_inscope"]),
        "three_class_macro_f1": bool(r["macro_f1"] >= LIMITS["macro_f1"]),
        "cancers_flagged": bool(r["cancer_flagged"] <= LIMITS["cancer_flagged"]),
        "calibration": bool(r["ece"] <= LIMITS["ece"]),
    }
    return all(checks.values()), checks


# ------------------------------------------------------------------ rows
def training_state(arm: str, tables: dict) -> dict:
    """What is learned from the training out-of-fold predictions only: the threshold and the temperature."""
    f = fuse(arm, "oof")
    y = lib.labels(tables["train"])
    ins = y < 3
    tau = threshold_for(f["other"][ins], FLAG_RATE)
    t = temperature(f["p3"][ins], y[ins])
    return {"fused": f, "y": y, "tau": tau, "temperature": t, "taus_info": {str(r): threshold_for(f["other"][ins], r) for r in OTHER_RATES}}


def sets_for(arm: str, tables: dict) -> tuple[dict, dict, dict, dict]:
    st = training_state(arm, tables)
    out = {}
    for role in ("val", "heldout"):
        f = fuse(arm, role)
        y = lib.labels(tables[role])
        out[role] = (f, y)
    fv, yv = out["val"]
    ins = yv < 3
    neg = {"other": fv["other"][ins], "p3": fv["p3"][ins], "y": yv[ins], "v1_p3": fv["v1_p3"][ins]}
    pos_held = {"other": out["heldout"][0]["other"]}
    pos_val = {"other": fv["other"][~ins]}
    return st, neg, pos_held, pos_val


def rehearse(arm: str, tables: dict) -> None:
    """The scoring machinery on out-of-fold training rows of fold 0, with everything learned from folds 1 to 4. No validation or held-out row."""
    f = fuse(arm, "oof")
    tr = tables["train"]
    y, fold = lib.labels(tr), tr.fold.values
    fit, ho = fold != 0, fold == 0
    ins_fit = fit & (y < 3)
    tau = threshold_for(f["other"][ins_fit], FLAG_RATE)
    t = temperature(f["p3"][ins_fit], y[ins_fit])
    neg_m, pos_m = ho & (y < 3), ho & (y == 3)
    neg = {"other": f["other"][neg_m], "p3": f["p3"][neg_m], "y": y[neg_m]}
    pos = {"other": f["other"][pos_m]}
    res = with_intervals(neg, pos, tau, t, tr.subject_id.values[neg_m], tr.subject_id.values[pos_m], b=200)
    ok, checks = adopt(res)
    print(f"REHEARSAL arm {arm} on fold 0 of the training out-of-fold predictions (not a result): tau {tau:.4f}, T {t:.2f}")
    for k, v in res.items():
        print(f"  {k:16} {v['value']:.3f} ({v['low']:.3f} to {v['high']:.3f})")
    print("  conditions:", checks, "-> would adopt" if ok else "-> would not adopt")


# ------------------------------------------------------------------ freeze, score
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_record() -> dict:
    choice = json.loads((OUT / "arm_choice.json").read_text(encoding="utf-8"))
    files = sorted(lib.PREDS.glob("*.npz"))
    return {"chosen_arm": choice["chosen"], "arm_choice": choice, "prediction_files": {f.name: sha(f) for f in files}}


def freeze() -> None:
    rec = frozen_record()
    (OUT / "frozen.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("frozen:", rec["chosen_arm"], len(rec["prediction_files"]), "prediction files")


def availability(tables: dict, role: str) -> pd.DataFrame:
    """Which inputs each row really had, by the app's rule: any lab value, a non-empty indication."""
    import train_text_baseline as tb
    t = tables[role]
    labs = t[list(lib.blood_mod.BLOOD_COLS)].notna().any(axis=1).values
    text = np.array([bool(tb.extract_indication(tb.read_report(p)).strip()) for p in t.report_path])
    return pd.DataFrame({"labs": labs, "indication": text})


def score() -> None:
    if (OUT / "results.json").exists():
        sys.exit("results.json exists: the validation and held-out rows were already scored once. A second look is no longer a test.")
    if not (OUT / "frozen.json").exists():
        sys.exit("run --freeze first")
    frozen = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    if frozen != json.loads(json.dumps(frozen_record())):
        sys.exit("the arm choice or a prediction file changed after --freeze")
    arm = frozen["chosen_arm"]
    tables = lib.load_tables()
    st, neg, pos_held, pos_val = sets_for(arm, tables)
    val_t, held_t = tables["val"], tables["heldout"]
    ins = lib.labels(val_t) < 3
    neg_subj, held_subj = val_t.subject_id.values[ins], held_t.subject_id.values
    tau, t = st["tau"], st["temperature"]
    result = with_intervals(neg, pos_held, tau, t, neg_subj, held_subj)
    adopted, checks = adopt(result)

    # the same statistic for the existing confidence signal on the same patients, paired through the bootstrap
    conf_neg = {"other": 1 - neg["v1_p3"].max(1), "p3": neg["v1_p3"], "y": neg["y"]}
    ho_f = fuse(arm, "heldout")
    conf_pos = {"other": 1 - ho_f["v1_p3"].max(1)}
    conf_tau = threshold_for(1 - st["fused"]["v1_p3"][st["y"] < 3].max(1), FLAG_RATE)
    rng = np.random.default_rng(SEED)
    gn, gp = groups_of(neg_subj), groups_of(held_subj)
    diffs = []
    for _ in range(B):
        iv, ih = draw(gn, rng), draw(gp, rng)
        a = statistics(neg, pos_held, tau, t, iv, ih)["auroc"]
        c = statistics(conf_neg, conf_pos, conf_tau, 1.0, iv, ih)["auroc"]
        diffs.append(a - c)
    conf_auroc = statistics(conf_neg, conf_pos, conf_tau, 1.0)["auroc"]

    # information: the Excluded studies of validation patients as a second set of positives, other flag rates, per class, availability
    neg_all = {"other": neg["other"], "y": neg["y"]}
    second = float(roc_auc_score(np.r_[np.zeros(len(neg["other"])), np.ones(len(pos_val["other"]))], np.r_[neg["other"], pos_val["other"]]))
    info = {"auroc_validation_excluded_as_positives": second,
            "flag_rates": {str(r): {"tau": st["taus_info"][str(r)], "flagged_inscope": float((neg["other"] >= st["taus_info"][str(r)]).mean()),
                                    "flagged_heldout": float((pos_held["other"] >= st["taus_info"][str(r)]).mean()),
                                    "flagged_validation_excluded": float((pos_val["other"] >= st["taus_info"][str(r)]).mean())}
                           for r in OTHER_RATES},
            "existing_confidence_signal": {"auroc": conf_auroc, "this_arm_minus_it": {"value": float(np.mean(diffs)),
                                           "low": float(np.percentile(diffs, 2.5)), "high": float(np.percentile(diffs, 97.5))}}}
    p, r, f, n = precision_recall_fscore_support(neg["y"], neg["p3"].argmax(1), labels=[0, 1, 2], zero_division=0)
    p1, r1, f1_, _ = precision_recall_fscore_support(neg["y"], neg["v1_p3"].argmax(1), labels=[0, 1, 2], zero_division=0)
    info["per_class_validation"] = {c: {"this_arm": {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])},
                                        "version_1": {"precision": float(p1[i]), "recall": float(r1[i]), "f1": float(f1_[i])}, "n": int(n[i])}
                                    for i, c in enumerate(lib.CLASSES[:3])}
    av_val, av_held = availability(tables, "val").loc[ins].reset_index(drop=True), availability(tables, "heldout")
    groups = {"all three inputs": lambda a: a.labs & a.indication, "no labs": lambda a: ~a.labs, "no indication": lambda a: ~a.indication}
    info["flag_rate_by_inputs"] = {k: {"inscope_validation": {"n": int(g(av_val).sum()), "flagged": float((neg["other"][g(av_val).values] >= tau).mean())},
                                       "heldout": {"n": int(g(av_held).sum()), "flagged": float((pos_held["other"][g(av_held).values] >= tau).mean())}}
                                   for k, g in groups.items()}
    summary = json.loads((OUT / "oof_summary.json").read_text(encoding="utf-8"))
    out = {"arm": arm, "operating_point": {"flag_rate_of_in_scope_training": FLAG_RATE, "tau": tau, "temperature": t},
           "limits": LIMITS, "statistics": result, "conditions": checks, "adopted": adopted, "information": info,
           "image_seed_spread": {k: v for k, v in summary["oof_information"].items() if k.startswith("image_seed")},
           "sizes": {"in_scope_validation": int(len(neg["y"])), "validation_patients": int(len(set(neg_subj))),
                     "heldout": int(len(pos_held["other"])), "heldout_patients": int(len(set(held_subj))),
                     "validation_excluded": int(len(pos_val["other"]))}}
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("arm", "statistics", "conditions", "adopted")}, indent=1))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearse", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.rehearse:
        tables = lib.load_tables()
        for arm in ("J", "T"):
            rehearse(arm, tables)
    elif args.freeze:
        freeze()
    else:
        score()


if __name__ == "__main__":
    main()
