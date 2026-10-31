"""
The scope check: does this case look like something the system has no class for?

The three class probabilities carry little information about that (experiments/other_pathology/RESULTS.md: the pooled confidence tells other
disease from in-scope disease with an AUROC of 0.62). The image backbone's own outputs carry more, because it was trained to report findings such
as effusion, edema, pneumothorax and cardiomegaly, which are exactly what make a study "other". The detector reads those, together with the three
models' probabilities, and returns a score.

It is a gate beside the other safeguards and, like them, it never changes the predicted class (the decision layer's rule: nothing here can emit
or alter a diagnosis). It needs an X-ray, because the backbone outputs come from the film; without one it says so and does nothing.

This file holds the feature definition, shared by the training script (scripts/train_out_of_scope.py) and the app, so a detector can never be
trained on one thing and served another. Plan and results: experiments/out_of_scope/.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RULES_PATH = ROOT / "app" / "data" / "out_of_scope_rules.json"
MODEL_PATH = ROOT / "experiments" / "out_of_scope" / "detector.joblib"

# The backbone has 18 outputs; seven are untrained slots that return a constant 0.5 for the weights in use (see docs/RESULTS.md), so
# only these eleven carry information: Atelectasis, Consolidation, Pneumothorax, Edema, Effusion, Pneumonia, Cardiomegaly, Lung Lesion,
# Fracture, Lung Opacity, Enlarged Cardiomediastinum.
INFORMATIVE18 = [0, 1, 3, 4, 7, 8, 10, 14, 15, 16, 17]
EPS = 1e-6


def pooled(image_p: np.ndarray, blood_p: np.ndarray, text_p: np.ndarray, has_blood: np.ndarray, has_text: np.ndarray) -> np.ndarray:
    """The app's rule (normalised geometric mean) over the inputs a case has. The image is always present here, because the check needs it."""
    num = np.log(np.clip(image_p, EPS, 1.0))
    den = np.ones(len(image_p))
    for p, h in ((blood_p, has_blood), (text_p, has_text)):
        w = np.asarray(h, dtype=float)
        num = num + w[:, None] * np.log(np.clip(p, EPS, 1.0))
        den = den + w
    g = np.exp(num / den[:, None])
    return g / g.sum(axis=1, keepdims=True)


def features(xrv18: np.ndarray, image_p: np.ndarray, blood_p: np.ndarray, text_p: np.ndarray, has_blood, has_text,
             with_backbone: bool = True) -> np.ndarray:
    """One row per case. Arm A of the plan uses with_backbone=False (what the system's class probabilities already know); arms B and C add
    the backbone's eleven informative outputs."""
    xrv18, image_p, blood_p, text_p = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (xrv18, image_p, blood_p, text_p))
    hb, ht = np.atleast_1d(np.asarray(has_blood, dtype=float)), np.atleast_1d(np.asarray(has_text, dtype=float))
    p = pooled(image_p, blood_p, text_p, hb, ht)
    entropy = -(p * np.log(np.clip(p, EPS, 1.0))).sum(axis=1)
    cols = [image_p, blood_p, text_p, hb[:, None], ht[:, None], p.max(axis=1)[:, None], entropy[:, None]]
    if with_backbone:
        cols.append(xrv18[:, INFORMATIVE18])
    return np.hstack(cols)


def feature_names(with_backbone: bool = True) -> list[str]:
    cls = ["normal", "pneumonia", "lung_cancer"]
    names = [f"{m}_p_{c}" for m in ("image", "blood", "text") for c in cls] + ["has_blood", "has_text", "pooled_max", "pooled_entropy"]
    if with_backbone:
        names += [f"xrv_{i}" for i in INFORMATIVE18]
    return names


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> dict | None:
    f = Path(path) if path else RULES_PATH
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@lru_cache(maxsize=1)
def _model():
    import joblib
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def available() -> bool:
    """True only when a fitted detector and its rules exist. None ships today (experiments/out_of_scope/RESULTS.md: the detector did not meet
    its adoption rule), so the check is dormant and a case pays nothing for it."""
    return load_rules() is not None and _model() is not None


def assess(xrv18, image_p, blood_p, text_p, has_blood: bool, has_text: bool, rules: dict | None = None, model=None) -> dict | None:
    """Score one case. Returns None when there is no fitted detector (the system then says nothing about scope). The result describes the case
    and never changes the prediction: {out_of_scope, score, threshold, flags_about}."""
    rules = rules if rules is not None else load_rules()
    model = model if model is not None else _model()
    if not rules or model is None:
        return None
    x = features(xrv18, image_p, blood_p, text_p, [has_blood], [has_text], with_backbone=rules["with_backbone"])
    score = float(model.predict_proba(x)[0, 1])
    return {"out_of_scope": bool(score >= rules["threshold"]), "score": round(score, 4), "threshold": rules["threshold"],
            "flag_rate_in_scope": rules["operating_point"]["flag_rate_in_scope"],
            "catch_rate_other": rules["operating_point"]["catch_rate_held_out"]}
