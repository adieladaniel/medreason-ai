"""
How sure the system is, and what it cannot rule out, from the pooled probabilities and the rules in
app/data/decision_rules.json (built by scripts/build_decision_rules.py from the training predictions).

Three things come out, and none of them is a diagnosis or changes the predicted class:

  low_confidence   the calibrated top probability is below the level where the reliability analysis declines to answer
  prediction_set   every class that cannot be ruled out, so that for each true class the set held it about 80 percent
                   (or 90) of the time on validation cases. One class means the inputs settle it, three means they do not.
  cancer_alert     "possible lung cancer": the pooled Lung Cancer probability reached the alert level. Averaging the models
                   loses cancers the text model finds, so this is a separate signal, not a replacement for the answer. It is
                   offered only when its alerts were at least MIN_ALERT_LIFT times more often real than a random case on
                   validation data (the image alone fails that: its alerts were barely better than chance).

The rules are calibrated for the exact combination of inputs that ran, because the pooled probabilities behave differently
for three inputs than for two, so a missing input selects a different rule set. Without the rules file, or for a
combination it does not cover, decide() returns None and the page shows nothing extra.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.agent import LABEL_NAMES

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "decision_rules.json"
# An alert must beat the base rate by this factor on validation cases to be shown. 2 is a judgement: it is clearly better than
# chance, and it removes the image-only alert (1.2 times) while keeping every combination that includes the text model.
MIN_ALERT_LIFT = 2.0
CANCER = LABEL_NAMES.index("Lung Cancer")


@lru_cache(maxsize=8)
def _read(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_rules(path: str | Path | None = None) -> dict | None:
    return _read(str(path or RULES_PATH))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    """Same function as scripts/eval_uncertainty.py: softmax of log p divided by the temperature."""
    z = np.log(p.clip(1e-9)) / t
    z = np.exp(z - z.max())
    return z / z.sum()


def decide(probabilities: dict, inputs, rules: dict | None = None, set_coverage: float = 0.8,
           alert_recall: float = 0.6) -> dict | None:
    """probabilities: the pooled belief {class: p}. inputs: the modalities that ran. Never reads or changes the prediction."""
    rules = rules if rules is not None else load_rules()
    if not rules:
        return None
    r = rules["by_inputs"].get("+".join(sorted(inputs)))
    cov, rec = f"{set_coverage:g}", f"{alert_recall:g}"
    if r is None or cov not in r["conformal"] or rec not in r["cancer_alert"]:
        return None

    p = np.array([probabilities[c] for c in LABEL_NAMES], dtype=float)
    confidence = float(apply_temperature(p, r["temperature"]).max())
    cut = r["confidence"]["threshold"]

    q = np.array(r["conformal"][cov]["quantiles"])
    members = [i for i in np.argsort(-p) if (1 - p[i]) <= q[i]]

    alert = r["cancer_alert"][rec]
    prevalence = rules["validation"]["cancer"] / rules["validation"]["cases"]
    lift = alert["validation"]["share_of_alerts_real"] / prevalence
    return {
        "rules_built": rules.get("built"),
        "low_confidence": confidence < cut,
        "confidence": {"calibrated": round(confidence, 4), "threshold": cut,
                       "answered": r["confidence"]["validation"]["answered"],
                       "accuracy_answered": r["confidence"]["validation"]["accuracy_answered"]},
        "prediction_set": [LABEL_NAMES[i] for i in members],
        "set_coverage": float(set_coverage),
        "set_validation": r["conformal"][cov]["validation"],
        "cancer_alert": {
            "informative": bool(lift >= MIN_ALERT_LIFT), "lift": round(float(lift), 2),
            "flag": bool(p[CANCER] >= alert["threshold"]), "probability": round(float(p[CANCER]), 4),
            "threshold": alert["threshold"], "target_recall": float(alert_recall), "validation": alert["validation"],
        },
    }
