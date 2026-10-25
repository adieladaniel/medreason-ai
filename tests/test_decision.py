"""How sure the system is: the rules file, applying it, building it, and the guarantee that it never touches the prediction."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from app.agent import decision
from app.agent.decision import decide, load_rules
from app.config import Settings

INPUTS = ["text", "blood", "image"]
CONFIDENT = {"Normal": 0.04, "Pneumonia": 0.92, "Lung Cancer": 0.04}
HARD = {"Normal": 0.38, "Pneumonia": 0.40, "Lung Cancer": 0.22}          # the validation lung cancer case the models get wrong
CANCER_SIGNAL = {"Normal": 0.18, "Pneumonia": 0.46, "Lung Cancer": 0.36}


# ------------------------------------------------------------------ the committed rules file
def test_the_rules_cover_every_combination_of_inputs():
    rules = load_rules()
    assert set(rules["by_inputs"]) == {"image", "blood", "text", "blood+image", "image+text", "blood+text", "blood+image+text"}
    assert rules["train"]["cancer"] > 0 and rules["validation"]["cancer"] > 0 and rules["set_coverages"] == [0.8, 0.9]


def test_the_rules_are_internally_consistent():
    for key, r in load_rules()["by_inputs"].items():
        assert 0 < r["temperature"] < 5 and 0 < r["confidence"]["threshold"] < 1, key
        for c in range(3):                                           # a 90 percent set is never stricter than an 80 percent one
            assert r["conformal"]["0.9"]["quantiles"][c] >= r["conformal"]["0.8"]["quantiles"][c], key
        ts = [r["cancer_alert"][k]["threshold"] for k in ("0.5", "0.6", "0.7", "0.8")]
        assert ts == sorted(ts, reverse=True), key                   # catching more cancers needs a lower bar
        recalls = [r["cancer_alert"][k]["validation"]["recall"] for k in ("0.5", "0.6", "0.7", "0.8")]
        assert recalls == sorted(recalls), key
        assert all(0 <= v <= 1 for k in ("0.5", "0.8") for v in (r["cancer_alert"][k]["validation"]["recall"], r["cancer_alert"][k]["validation"]["share_of_alerts_real"]))


def test_the_all_inputs_rules_match_the_reliability_analysis():
    """Same method as scripts/eval_uncertainty.py, so the headline numbers on the Reliability page must reappear here."""
    r = load_rules()["by_inputs"]["blood+image+text"]
    ref = json.loads(Path("experiments/uncertainty/metrics.json").read_text(encoding="utf-8"))
    assert r["temperature"] == pytest.approx(ref["calibration"]["temperature"], abs=0.001)
    assert r["confidence"]["threshold"] == pytest.approx(ref["abstain_rule"]["threshold"], abs=0.001)
    assert r["conformal"]["0.8"]["quantiles"] == pytest.approx(ref["conformal"]["0.2"]["quantiles"], abs=0.001)
    assert r["conformal"]["0.9"]["quantiles"] == pytest.approx(ref["conformal"]["0.1"]["quantiles"], abs=0.001)


# ------------------------------------------------------------------ applying them
def test_a_confident_case_is_answered_with_a_one_class_set_and_no_alert():
    d = decide(CONFIDENT, INPUTS)
    assert d["low_confidence"] is False and d["prediction_set"] == ["Pneumonia"] and d["cancer_alert"]["flag"] is False


def test_the_hard_case_is_low_confidence_and_nothing_can_be_ruled_out():
    d = decide(HARD, INPUTS)
    assert d["low_confidence"] is True and d["confidence"]["calibrated"] < d["confidence"]["threshold"]
    assert d["prediction_set"] == ["Pneumonia", "Normal", "Lung Cancer"]            # most likely first
    assert d["cancer_alert"]["flag"] is False                                      # 0.22 is under the default level, said plainly


def test_a_cancer_signal_under_another_headline_raises_the_alert():
    d = decide(CANCER_SIGNAL, INPUTS)
    a = d["cancer_alert"]
    assert a["flag"] is True and a["informative"] is True and a["probability"] >= a["threshold"]
    assert "Lung Cancer" in d["prediction_set"]


def test_a_more_sensitive_alert_level_flags_the_hard_case():
    assert decide(HARD, INPUTS, alert_recall=0.8)["cancer_alert"]["flag"] is True
    assert decide(HARD, INPUTS, alert_recall=0.6)["cancer_alert"]["flag"] is False


def test_a_higher_set_coverage_never_shrinks_the_set():
    for p in (CONFIDENT, HARD, CANCER_SIGNAL):
        assert set(decide(p, INPUTS, set_coverage=0.8)["prediction_set"]) <= set(decide(p, INPUTS, set_coverage=0.9)["prediction_set"])


def test_the_rules_used_depend_on_which_inputs_ran_not_on_their_order():
    p = {"Normal": 0.93, "Pneumonia": 0.05, "Lung Cancer": 0.02}
    assert decide(p, ["image", "text"]) == decide(p, ["text", "image"])
    assert decide(p, ["image", "text"])["confidence"]["threshold"] != decide(p, INPUTS)["confidence"]["threshold"]


def test_an_alert_that_is_barely_better_than_chance_is_not_offered():
    """The image alone flagged 42 of every 100 validation cases and only 6 percent were cancer, against a base rate near 5."""
    a = decide(CANCER_SIGNAL, ["image"])["cancer_alert"]
    assert a["informative"] is False and a["lift"] < decision.MIN_ALERT_LIFT
    assert decide(CANCER_SIGNAL, INPUTS)["cancer_alert"]["informative"] is True


def test_nothing_is_returned_without_rules_or_for_an_unknown_combination(tmp_path):
    assert decide(CONFIDENT, INPUTS, rules={}) is None
    assert decide(CONFIDENT, ["audio"]) is None
    assert load_rules(tmp_path / "missing.json") is None
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert load_rules(tmp_path / "broken.json") is None


def test_it_never_changes_its_input_and_never_names_a_class_of_its_own():
    """Design rule: the models diagnose, this only says how sure. It has no predicted class to change."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.dirichlet([0.6, 0.6, 0.6])
        probs = {"Normal": float(p[0]), "Pneumonia": float(p[1]), "Lung Cancer": float(p[2])}
        before = copy.deepcopy(probs)
        d = decide(probs, INPUTS)
        assert probs == before and "predicted_class" not in d
        assert set(d["prediction_set"]) <= set(probs)


def test_a_set_can_be_empty_and_that_is_reported_not_hidden():
    rules = copy.deepcopy(load_rules())
    rules["by_inputs"]["text"]["conformal"]["0.8"]["quantiles"] = [0.01, 0.01, 0.01]
    d = decide({"Normal": 0.4, "Pneumonia": 0.4, "Lung Cancer": 0.2}, ["text"], rules=rules)
    assert d["prediction_set"] == []


# ------------------------------------------------------------------ settings
def test_the_settings_accept_only_the_levels_the_rules_were_built_for():
    assert Settings().prediction_set_coverage == 0.8 and Settings().cancer_alert_recall == 0.6
    with pytest.raises(ValueError, match="0.8 or 0.9"):
        Settings(prediction_set_coverage=0.85)
    with pytest.raises(ValueError, match="0.5, 0.6, 0.7 or 0.8"):
        Settings(cancer_alert_recall=0.95)


# ------------------------------------------------------------------ building them
def synthetic(n, rng, strength=1.3):
    """Three models that are each right about as often as a real, imperfect model, with 8 percent rare-class cases."""
    y = rng.choice(3, size=n, p=[0.35, 0.57, 0.08])
    blocks = []
    for _ in range(3):
        z = rng.normal(0, 1, (n, 3)) + strength * np.eye(3)[y]
        e = np.exp(z - z.max(1, keepdims=True))
        blocks.append(e / e.sum(1, keepdims=True))
    return np.concatenate(blocks, axis=1), y


def test_the_builder_gives_rules_that_hold_on_cases_it_did_not_see():
    from scripts.build_decision_rules import build_rules
    rng = np.random.default_rng(3)
    (Xtr, ytr), (Xva, yva) = synthetic(6000, rng), synthetic(4000, rng)
    rules = build_rules(Xtr, ytr, Xva, yva)
    assert set(rules["by_inputs"]) == set(load_rules()["by_inputs"])
    r = rules["by_inputs"]["blood+image+text"]
    for c, name in enumerate(["Normal", "Pneumonia", "Lung Cancer"]):                    # coverage target is met for every class
        assert r["conformal"]["0.8"]["validation"]["coverage"][name] >= 0.8 - 0.06, name
        assert r["conformal"]["0.9"]["validation"]["coverage"][name] >= 0.9 - 0.06, name
    assert r["cancer_alert"]["0.8"]["validation"]["recall"] > r["cancer_alert"]["0.5"]["validation"]["recall"]
    assert 0.74 < r["confidence"]["validation"]["answered"] < 0.86                        # keeps about 80 percent, set on train
    assert r["confidence"]["validation"]["accuracy_answered"] >= r["confidence"]["validation"]["accuracy_all"]


@pytest.mark.slow
@pytest.mark.skipif(not Path("experiments/blood_baseline/oof_train_predictions.csv").exists(), reason="needs the local training predictions")
def test_the_committed_rules_are_what_the_training_predictions_produce():
    from scripts.build_decision_rules import build_rules
    from scripts.train_fusion import assemble
    _, Xtr, ytr = assemble("train")
    _, Xva, yva = assemble("val")
    fresh, saved = build_rules(Xtr, ytr, Xva, yva), load_rules()
    for k in ("built",):
        fresh.pop(k), saved.pop(k)
    assert fresh == saved
