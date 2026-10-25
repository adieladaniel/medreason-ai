"""
Builds app/data/decision_rules.json: the small file the app reads to say how sure it is, without needing the
training predictions (which are git-ignored and absent from a deployment).

Three things are derived, each for every combination of inputs that can occur (a case may lack the image, the labs
or the indication, and the pooled probabilities behave differently for each, so a rule calibrated on all three would be
unjustified for two):

1. Confidence. Temperature scaling (one number) turns the pooled probabilities into calibrated ones, and the case is
   called low confidence when its calibrated top probability falls below the level that keeps 80 percent of the TRAIN
   cases. This is the abstain rule from scripts/eval_uncertainty.py (0.55 for all three inputs).
2. Prediction sets. Class-conditional (Mondrian) split conformal, the same method and functions as
   eval_uncertainty.py: a class is in the set when it cannot be ruled out, so that for every true class the set holds it
   about 80 percent (or 90 percent) of the time.
3. A cancer alert. Flag "possible lung cancer" when the pooled Lung Cancer probability is at least t. Averaging the
   models loses cancers the text model finds (50 percent found against 61 percent), so t is picked on TRAIN to catch a
   target share of the true cancers there, at four levels.

Everything is fitted on the out-of-fold training predictions only. The validation split is used once, to report what each
rule does on cases it has not seen, and those numbers are stored next to the rule so the page can quote them. Nothing
here is tuned on validation. The test split stays sealed.

Run from the repo root:  python scripts/build_decision_rules.py
"""
import torch  # noqa: F401  (Windows: torch before sklearn, see CLAUDE.md gotcha 7)

import json
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scripts.eval_uncertainty import apply_temperature, conformal_quantiles, fit_temperature, pooled, prediction_sets  # noqa: E402
from scripts.train_fusion import LABEL_NAMES, SOURCES, assemble  # noqa: E402

OUT = Path("app/data/decision_rules.json")
INPUTS = list(SOURCES)                                  # image, blood, text: the order of the fusion feature blocks
SET_COVERAGES = (0.8, 0.9)
ALERT_RECALLS = (0.5, 0.6, 0.7, 0.8)
KEEP_ON_TRAIN = 0.8                                     # share of train cases the confidence rule answers
CANCER = 2


def input_combinations():
    """Every non-empty subset of the three inputs, as (key, column indices). The key is the sorted names joined by '+'."""
    for r in range(1, len(INPUTS) + 1):
        for idx in combinations(range(len(INPUTS)), r):
            yield "+".join(sorted(INPUTS[i] for i in idx)), idx


def rules_for(Ptr, ytr, Pva, yva) -> dict:
    """Rules fitted on (Ptr, ytr), pooled probabilities of the train out-of-fold cases, and what they do on (Pva, yva)."""
    T = fit_temperature(Ptr, ytr)
    tau = float(np.quantile(apply_temperature(Ptr, T).max(1), 1 - KEEP_ON_TRAIN))
    keep = apply_temperature(Pva, T).max(1) >= tau
    pred = Pva.argmax(1)
    out = {
        "temperature": round(T, 4),
        "confidence": {"threshold": round(tau, 4), "keeps_on_train": KEEP_ON_TRAIN,
                       "validation": {"answered": round(float(keep.mean()), 4),
                                      "accuracy_all": round(float((pred == yva).mean()), 4),
                                      "accuracy_answered": round(float((pred[keep] == yva[keep]).mean()), 4)}},
        "conformal": {},
        "cancer_alert": {},
    }
    for cov in SET_COVERAGES:
        q = conformal_quantiles(Ptr, ytr, 1 - cov)
        sets = prediction_sets(Pva, q)
        size = sets.sum(1)
        out["conformal"][f"{cov:g}"] = {
            "quantiles": [round(float(v), 4) for v in q],
            "validation": {"coverage": {LABEL_NAMES[c]: round(float(sets[yva == c, c].mean()), 4) for c in range(3)},
                           "mean_size": round(float(size.mean()), 3), "share_single": round(float((size == 1).mean()), 4)}}
    for target in ALERT_RECALLS:
        t = float(np.quantile(Ptr[ytr == CANCER, CANCER], 1 - target))
        flag = Pva[:, CANCER] >= t
        caught = int((flag & (yva == CANCER)).sum())
        out["cancer_alert"][f"{target:g}"] = {
            "threshold": round(t, 4),
            "validation": {"recall": round(caught / max(int((yva == CANCER).sum()), 1), 4),
                           "alerts_per_100": round(100 * float(flag.mean()), 2),
                           "share_of_alerts_real": round(caught / max(int(flag.sum()), 1), 4)}}
    return out


def build_rules(Xtr, ytr, Xva, yva) -> dict:
    by_inputs = {}
    for key, idx in input_combinations():
        by_inputs[key] = rules_for(pooled(Xtr, idx), ytr, pooled(Xva, idx), yva)
    return {
        "built": date.today().isoformat(),
        "how": "scripts/build_decision_rules.py; fitted on train out-of-fold predictions, reported once on validation",
        "train": {"cases": int(len(ytr)), "cancer": int((ytr == CANCER).sum())},
        "validation": {"cases": int(len(yva)), "cancer": int((yva == CANCER).sum())},
        "set_coverages": list(SET_COVERAGES), "alert_recalls": list(ALERT_RECALLS),
        "by_inputs": by_inputs,
    }


def main() -> None:
    _, Xtr, ytr = assemble("train")
    _, Xva, yva = assemble("val")
    rules = build_rules(Xtr, ytr, Xva, yva)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rules, indent=1), encoding="utf-8")
    print(f"wrote {OUT} from {rules['train']['cases']} train cases ({rules['train']['cancer']} cancer), checked on {rules['validation']['cases']} validation cases\n")
    print(f"{'inputs':<20}{'conf cut':>9}{'answered':>10}{'acc.':>7}  {'80% set: coverage N/P/C':>26}{'single':>8}  {'alert 60%: caught':>18}{'per 100':>9}{'real':>7}")
    for key, r in rules["by_inputs"].items():
        c, s, a = r["confidence"], r["conformal"]["0.8"]["validation"], r["cancer_alert"]["0.6"]["validation"]
        cov = "/".join(f"{s['coverage'][k]:.2f}" for k in LABEL_NAMES)
        print(f"{key:<20}{r['confidence']['threshold']:>9.3f}{c['validation']['answered']:>10.0%}{c['validation']['accuracy_answered']:>7.0%}  {cov:>26}{s['share_single']:>8.0%}  {a['recall']:>18.0%}{a['alerts_per_100']:>9.1f}{a['share_of_alerts_real']:>7.0%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
