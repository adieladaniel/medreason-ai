"""
Consistency checker: flags when the modalities disagree about the predicted class.

The severity language it uses ("cases like this are right N% of the time") comes from
app/data/dashboard.json's predictions.agreement section, which is computed from real
validation predictions (app/dashboard_data.py, _agreement()). If that file is missing,
for example on a fresh clone before the dashboard has been built once, this falls back to
the same numbers written in as constants, sourced from the same computation, so the
checker never fabricates a confidence figure.

Pairwise ablation scores (image+text 0.557, blood+text 0.576, image+blood 0.493 macro F1,
from experiments/fusion/metrics.json) are cited for two-modality cases, where the
dashboard has no equivalent breakdown because it only ever runs all three.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.agent import LABEL_NAMES

DASHBOARD_JSON = Path(__file__).resolve().parent.parent / "data" / "dashboard.json"

# Fallback if dashboard.json has not been built yet. Source: app/dashboard_data.py
# _agreement(), computed from experiments/*/val_predictions.csv (val, n=2315).
_FALLBACK_LEVELS = [
    {"name": "All three agree", "share": 0.3127, "fused_accuracy": 0.8605},
    {"name": "Two agree", "share": 0.5931, "fused_accuracy": 0.6759},
    {"name": "All three differ", "share": 0.0942, "fused_accuracy": 0.5321},
]

_PAIR_MACRO_F1 = {
    frozenset(("image", "text")): 0.557,
    frozenset(("blood", "text")): 0.576,
    frozenset(("image", "blood")): 0.493,
}


def _agreement_levels() -> list[dict]:
    if DASHBOARD_JSON.exists():
        try:
            d = json.loads(DASHBOARD_JSON.read_text(encoding="utf-8"))
            return d["predictions"]["agreement"]["levels"]
        except (KeyError, ValueError):
            pass
    return _FALLBACK_LEVELS


def check_consistency(evidence: dict[str, np.ndarray]) -> dict:
    """evidence: {modality: 3-vector of probabilities}, whichever modalities have run."""
    preds = {m: LABEL_NAMES[int(p.argmax())] for m, p in evidence.items()}
    n = len(preds)
    distinct = set(preds.values())

    if n < 2:
        return {
            "status": "single_modality",
            "modality_predictions": preds,
            "detail": "Only one modality has run, so there is nothing yet to check for agreement.",
        }

    if n == 2:
        (m1, p1), (m2, p2) = preds.items()
        agree = p1 == p2
        pair_f1 = _PAIR_MACRO_F1.get(frozenset(preds.keys()))
        note = f" This pair reached macro F1 {pair_f1} on validation." if pair_f1 else ""
        return {
            "status": "agree" if agree else "conflict",
            "modality_predictions": preds,
            "detail": (
                f"{m1.capitalize()} and {m2.capitalize()} both point to {p1}.{note}"
                if agree else
                f"{m1.capitalize()} says {p1}, {m2.capitalize()} says {p2}.{note} "
                "A third modality would help decide between them."
            ),
        }

    # all three ran
    levels = _agreement_levels()
    top_count = max(list(preds.values()).count(v) for v in distinct)
    level = levels[0] if top_count == 3 else levels[1] if top_count == 2 else levels[2]
    status = "agree" if top_count == 3 else "partial" if top_count == 2 else "conflict"
    odd_one = None
    if status == "partial":
        odd_one = next(m for m, v in preds.items() if list(preds.values()).count(v) == 1)

    detail = f"{level['name']}."
    if odd_one:
        detail += f" {odd_one.capitalize()} is the outlier, predicting {preds[odd_one]}."
    detail += (
        f" On validation, cases where {level['name'].lower()} were correct "
        f"{level['fused_accuracy']:.0%} of the time ({level['share']:.0%} of cases look like this)."
    )
    return {"status": status, "modality_predictions": preds, "detail": detail}
