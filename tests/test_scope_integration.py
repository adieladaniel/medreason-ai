"""
The scope check inside the prediction stage (app/pipelines/staged.py: scope_of, stage_predict): what it is given, that an absent input is
filled the way training filled it, that nothing about the prediction depends on it, and (slow) that the features computed live equal the
features the detector was trained on.
"""
import numpy as np
import pytest

from app.agent import scope as scope_check
from app.pipelines import staged
from tests.conftest import probs
from tests.test_pipeline import prediction_of

XRV = np.linspace(0.05, 0.95, 18)
# asymmetric on purpose: a vector like (0.1, 0.8, 0.1) reads the same backwards, which would hide a class-order mistake
TABLE = {"image": probs(0.2, 0.7, 0.1), "blood": probs(0.5, 0.3, 0.2), "text": probs(0.1, 0.6, 0.3)}


def predictors(calls=None, with_outputs=True):
    def make(m):
        def f(inp):
            if calls is not None:
                calls.append((m, inp))
            empty = inp == "" or (isinstance(inp, dict) and all(v is None for v in inp.values()))
            r = prediction_of(m, probs(0.3, 0.4, 0.3) if empty else TABLE[m])           # a model asked about nothing gives its own constant
            if m == "image" and with_outputs:
                r["_backbone_outputs"] = XRV
            return r
        return f
    return {m: make(m) for m in ("image", "blood", "text")}


@pytest.fixture(autouse=True)
def fresh_cache():
    staged._MISSING_INPUT.clear()
    yield
    staged._MISSING_INPUT.clear()


@pytest.fixture
def captured(monkeypatch):
    seen = []

    def fake_assess(xrv18, image_p, blood_p, text_p, has_blood, has_text, **kw):
        seen.append(dict(xrv18=xrv18, image_p=image_p, blood_p=blood_p, text_p=text_p, has_blood=has_blood, has_text=has_text))
        return {"out_of_scope": True, "score": 0.9, "threshold": 0.4, "flag_rate_in_scope": 0.1, "catch_rate_other": 0.5}
    monkeypatch.setattr(scope_check, "assess", fake_assess)
    monkeypatch.setattr(scope_check, "available", lambda: True)
    return seen


def run(inputs, preds):
    return staged.stage_predict({"deps": {"predictors": preds}, "inputs": inputs})


def test_the_check_is_given_the_backbone_outputs_the_probabilities_in_class_order_and_which_inputs_are_real(captured):
    out = run({"image": "x.png", "blood": {"wbc": 9.0}, "text": "fever"}, predictors())
    (seen,) = captured
    assert np.array_equal(seen["xrv18"], XRV) and seen["has_blood"] is True and seen["has_text"] is True
    assert np.allclose(seen["image_p"], [0.2, 0.7, 0.1]) and np.allclose(seen["blood_p"], [0.5, 0.3, 0.2]) and np.allclose(seen["text_p"], [0.1, 0.6, 0.3])
    assert out["scope"]["out_of_scope"] is True and out["belief"]["predicted_class"] == "Pneumonia"


def test_an_absent_input_is_filled_with_what_its_model_says_about_nothing_as_in_training(captured):
    calls = []
    out = run({"image": "x.png"}, predictors(calls))
    (seen,) = captured
    assert seen["has_blood"] is False and seen["has_text"] is False
    from explain.blood import RAW_LABS
    assert ("blood", {k: None for k in RAW_LABS}) in calls and ("text", "") in calls            # the all-missing labs and the empty indication
    assert np.allclose(seen["blood_p"], [0.3, 0.4, 0.3]) and np.allclose(seen["text_p"], [0.3, 0.4, 0.3])
    assert out["modalities_used"] == ["image"]                                                   # filling for the check does not add an input


def test_the_fill_is_computed_once_per_modality_not_per_case(captured):
    calls = []
    preds = predictors(calls)
    run({"image": "x.png"}, preds)
    run({"image": "x.png"}, preds)
    assert sum(1 for m, _ in calls if m == "blood" and isinstance(_, dict) and all(v is None for v in _.values())) == 1
    assert sum(1 for m, a in calls if m == "text" and a == "") == 1


def test_without_a_film_the_check_does_not_run(captured):
    out = run({"text": "fever", "blood": {"wbc": 9.0}}, predictors())
    assert out["scope"] is None and captured == []


def test_without_the_backbone_outputs_the_check_does_not_run(captured):
    out = run({"image": "x.png", "text": "fever"}, predictors(with_outputs=False))
    assert out["scope"] is None and captured == []


def test_with_no_detector_a_case_pays_nothing_for_the_check(monkeypatch):
    calls = []
    monkeypatch.setattr(scope_check, "available", lambda: False)
    monkeypatch.setattr(scope_check, "assess", lambda *a, **k: calls.append(1))
    out = run({"image": "x.png"}, predictors(calls))              # image only, so the filler for blood and text would be needed
    assert out["scope"] is None and calls == [("image", "x.png")]  # no assess call and no extra "model asked about nothing" calls
    assert staged._MISSING_INPUT == {}


def test_without_a_fitted_detector_the_prediction_carries_no_scope_and_is_otherwise_the_same(monkeypatch):
    monkeypatch.setattr(scope_check, "load_rules", lambda path=None: None)
    monkeypatch.setattr(scope_check, "_model", lambda: None)
    out = run({"image": "x.png", "blood": {"wbc": 9.0}, "text": "fever"}, predictors())
    assert out["scope"] is None and out["belief"]["predicted_class"] == "Pneumonia"


def test_the_check_cannot_change_anything_else_in_the_prediction(monkeypatch, captured):
    inputs = {"image": "x.png", "blood": {"wbc": 9.0}, "text": "fever"}
    flagged = run(inputs, predictors())
    monkeypatch.setattr(scope_check, "assess", lambda *a, **k: None)
    plain = run(inputs, predictors())
    assert flagged["scope"] is not None and plain["scope"] is None
    for key in ("belief", "consistency", "missing_info_plan", "decision", "modalities", "modalities_used"):
        assert flagged[key] == plain[key], key


def test_the_stored_scope_is_plain_json_and_the_private_outputs_are_not_stored(captured):
    import json
    out = run({"image": "x.png", "text": "fever"}, predictors())
    json.dumps(out)                                                              # no numpy arrays anywhere in what is stored
    assert "_backbone_outputs" not in json.dumps(out)


@pytest.mark.slow
def test_features_computed_live_equal_the_features_the_detector_was_trained_on():
    """The train/serve consistency check: for validation studies, the live image, blood and text predictions give the same detector features as
    the cached backbone outputs and stored prediction files that training used."""
    import pandas as pd
    from app.agent.tools import PREDICTORS
    from app.agent.orchestrator import _case_from_split
    from app.pipelines.staged import provided_inputs
    from scripts.eval_test_split import case_inputs
    from explain.blood import RAW_LABS

    val = pd.read_csv("data/processed/val.csv")
    cached = np.load("experiments/image_baseline/feats_val.npy")
    cols = {"image": ("image", "image_p_"), "blood": ("blood", "blood_p_"), "text": ("text", "text_indication_p_")}
    stored = {m: pd.read_csv(f"experiments/{d}_baseline/val_predictions.csv").set_index("study_id") for m, (d, _) in cols.items()}
    slug = ["normal", "pneumonia", "lung_cancer"]
    for idx in (0, 7, 31, 120, 800):
        row = val.iloc[idx]
        c = case_inputs(row)
        have = provided_inputs(c["indication"], c["labs"], c["image"])
        img = PREDICTORS["image"](c["image"])
        live = scope_check.features(img["_backbone_outputs"], [img["probabilities"][k] for k in ("Normal", "Pneumonia", "Lung Cancer")],
                                    [PREDICTORS["blood"](c["labs"])["probabilities"][k] for k in ("Normal", "Pneumonia", "Lung Cancer")],
                                    [PREDICTORS["text"](c["indication"] or "")["probabilities"][k] for k in ("Normal", "Pneumonia", "Lung Cancer")],
                                    ["blood" in have], ["text" in have])
        sid = row["study_id"]
        train_side = scope_check.features(cached[idx], stored["image"].loc[sid, ["image_p_" + s for s in slug]].values,
                                          stored["blood"].loc[sid, ["blood_p_" + s for s in slug]].values,
                                          stored["text"].loc[sid, ["text_indication_p_" + s for s in slug]].values,
                                          [bool(row[RAW_LABS].notna().any())], ["text" in have])
        assert np.abs(live - train_side).max() < 2e-3, idx
