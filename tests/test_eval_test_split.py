"""
scripts/eval_test_split.py: the one evaluation on the sealed split. The safeguards are tested hardest, because a mistake here cannot be
undone: the prediction step must never load the labels, scoring must refuse to run twice, and a change to the system between the
two steps must stop the run. The arithmetic is tested on synthetic predictions, with the real decision rules.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from app.agent.belief import BeliefState
from app.agent.decision import load_rules
from scripts import eval_test_split as ev

CLASSES = ev.CLASSES


def fake_probs(rng, n, skill=1.5):
    y = rng.choice(3, n, p=[0.3, 0.6, 0.1])
    P = {}
    for m in ev.MODS:
        logits = rng.normal(0, 1, (n, 3)) + skill * np.eye(3)[y]
        e = np.exp(logits)
        P[m] = e / e.sum(1, keepdims=True)
    return y, P


# ------------------------------------------------------------------ pooling
def test_pooling_with_a_mask_leaves_out_absent_models_and_without_one_uses_all_three():
    rng = np.random.default_rng(0)
    y, P = fake_probs(rng, 20)
    mask = {m: np.ones(20, bool) for m in ev.MODS}
    mask["blood"][:5] = False
    deployed, convention = ev.pool(P, mask), ev.pool(P, None)
    for i in range(20):
        b = BeliefState()
        for m in ev.MODS:
            if m != "blood" or i >= 5:
                b.update(m, dict(zip(CLASSES, P[m][i])))
        assert np.allclose(deployed[i], b.pool(), atol=1e-9)
    assert np.allclose(deployed[5:], convention[5:]) and not np.allclose(deployed[:5], convention[:5])
    assert np.allclose(deployed.sum(1), 1)


def test_a_uniform_vector_does_not_change_the_winning_class_but_flattens_the_probabilities():
    # why a failed model is filled with 1/3 for the all-three convention: the class is what it would have been without that model
    rng = np.random.default_rng(1)
    _, P = fake_probs(rng, 200)
    Q = dict(P)
    Q["blood"] = np.full((200, 3), 1 / 3)
    with_uniform, without = ev.pool(Q, None), ev.pool(P, None, ("image", "text"))
    assert (with_uniform.argmax(1) == without.argmax(1)).all()
    assert with_uniform.max(1).mean() < without.max(1).mean()


# ------------------------------------------------------------------ the arithmetic
def test_perfect_predictions_score_perfectly_and_the_two_conventions_agree_when_every_input_is_present():
    rng = np.random.default_rng(2)
    y = rng.choice(3, 120, p=[0.3, 0.5, 0.2])
    P = {m: np.eye(3)[y] * 0.98 + 0.01 for m in ev.MODS}
    mask = {m: np.ones(120, bool) for m in ev.MODS}
    r = ev.evaluate(P, mask, y, np.arange(120), load_rules(), b=20)
    for k in ("deployed", "validation_convention"):
        assert r[k]["point"]["macro_f1"] == 1.0 and r[k]["point"]["accuracy"] == 1.0 and r[k]["point"]["auc_cancer"] == 1.0
    assert r["deployed"]["point"] == r["validation_convention"]["point"]
    assert r["majority_class_accuracy"] == pytest.approx(np.bincount(y).max() / 120)


def test_the_result_has_every_part_the_plan_promises():
    rng = np.random.default_rng(3)
    y, P = fake_probs(rng, 240)
    mask = {m: rng.random(240) > 0.1 for m in ev.MODS}
    mask["image"][:] = True
    r = ev.evaluate(P, mask, y, rng.integers(0, 80, 240), load_rules(), b=40)
    assert set(r) >= {"n", "n_cancer", "majority_class_accuracy", "inputs_present", "deployed", "validation_convention"}
    for name in ("deployed", "validation_convention"):
        p, i = r[name]["point"], r[name]["interval"]
        for k in ("macro_f1", "f1_cancer", "accuracy", "auc_cancer", "ece", "brier", "ece_calibrated", "answered_share", "accuracy_answered",
                  "cancers_declined", "set_coverage_lung_cancer", "single_class_set", "image_macro_f1", "blood_auc_cancer",
                  "alert_0.6_recall", "alert_0.8_per_100", "alert_0.5_share_real"):
            assert k in p and k in i
        assert np.array(r[name]["confusion"]).sum() == 240
        assert sum(v["n"] for v in r[name]["consistency"].values()) == 240
        assert set(r[name]["pairs_macro_f1"]) == {"image+blood", "image+text", "blood+text", "image+blood+text"}


def test_a_higher_alert_level_flags_at_least_as_many_cancers():
    rng = np.random.default_rng(4)
    y, P = fake_probs(rng, 400, skill=1.0)
    mask = {m: np.ones(400, bool) for m in ev.MODS}
    p = ev.evaluate(P, mask, y, np.arange(400), load_rules(), b=10)["validation_convention"]["point"]
    recalls = [p[f"alert_{lv:g}_recall"] for lv in ev.ALERT_LEVELS]
    per100 = [p[f"alert_{lv:g}_per_100"] for lv in ev.ALERT_LEVELS]
    assert recalls == sorted(recalls) and per100 == sorted(per100)


def test_the_bootstrap_resamples_patients_not_studies():
    # 40 patients with 5 identical studies each: treated as 200 independent studies the interval is too narrow, as 40 patients it is wider
    rng = np.random.default_rng(5)
    y0, P0 = fake_probs(rng, 40)
    y = np.repeat(y0, 5)
    P = {m: np.repeat(P0[m], 5, axis=0) for m in ev.MODS}
    mask = {m: np.ones(200, bool) for m in ev.MODS}
    as_studies = ev.evaluate(P, mask, y, np.arange(200), load_rules(), b=150)["deployed"]["interval"]["accuracy"]
    as_patients = ev.evaluate(P, mask, y, np.repeat(np.arange(40), 5), load_rules(), b=150)["deployed"]["interval"]["accuracy"]
    assert (as_patients[1] - as_patients[0]) > 1.5 * (as_studies[1] - as_studies[0])
    assert ev.evaluate(P, mask, y, np.arange(200), load_rules(), b=30) == ev.evaluate(P, mask, y, np.arange(200), load_rules(), b=30)


# ------------------------------------------------------------------ the safeguards
def test_the_prediction_step_never_loads_the_label_column(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({"study_id": [1], "subject_id": [2], "final_label": ["Normal"], "image_path": ["x"]}).to_csv(f, index=False)
    assert "final_label" not in ev.load_inputs(f).columns
    for fn in (ev.predict_row, ev.case_inputs, ev.run_predict):
        assert "final_label" not in inspect.getsource(fn), fn.__name__
    assert inspect.getsource(ev.load_inputs).count("final_label") == 1                       # only to say it is left out


def test_scoring_refuses_to_run_twice_and_does_not_touch_the_labels_when_it_refuses(tmp_path, monkeypatch):
    done = tmp_path / "results.json"
    done.write_text("{}")
    monkeypatch.setattr(ev, "RESULTS", done)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a file was read")))
    with pytest.raises(SystemExit) as e:
        ev.run_score()
    assert "second look" in str(e.value)


def test_scoring_needs_the_predictions_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "none.json")
    monkeypatch.setattr(ev, "PREDICTIONS", tmp_path / "none.csv")
    monkeypatch.setattr(ev, "FROZEN", tmp_path / "none.frozen")
    with pytest.raises(SystemExit) as e:
        ev.run_score()
    assert "--predict" in str(e.value)


@pytest.fixture
def fake_system(tmp_path, monkeypatch):
    """The frozen files as small stand-ins in a scratch folder: the real models are git-ignored, so a clean checkout (CI) does not have them."""
    root = tmp_path / "repo"
    for f in ev.FROZEN_FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(f"contents of {f}")
    backbone = tmp_path / "backbone.pt"
    backbone.write_text("weights")
    monkeypatch.setattr(ev, "ROOT", root)
    monkeypatch.setattr(ev, "BACKBONE", backbone)
    return root


def test_a_change_to_the_system_between_predicting_and_scoring_stops_the_run(tmp_path, monkeypatch, fake_system):
    state = ev.frozen_system()
    frozen, pred = tmp_path / "f.json", tmp_path / "p.csv"
    frozen.write_text(json.dumps(state))
    pred.write_text("study_id\n1\n")
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "r.json")
    monkeypatch.setattr(ev, "FROZEN", frozen)
    monkeypatch.setattr(ev, "PREDICTIONS", pred)
    changed = json.loads(json.dumps(state))
    changed["files"]["app/data/decision_rules.json"] = "0" * 64
    monkeypatch.setattr(ev, "frozen_system", lambda: changed)
    with pytest.raises(SystemExit) as e:
        ev.run_score()
    assert "changed" in str(e.value)
    with pytest.raises(SystemExit) as e2:
        ev.run_predict()
    assert "changed since the first prediction run" in str(e2.value)


def test_the_frozen_record_covers_the_models_the_rules_and_the_prediction_path_but_not_the_scoring_script(fake_system):
    s = ev.frozen_system()
    files = set(s["files"])
    assert {"experiments/blood_baseline/hgb_model.joblib", "experiments/text_baseline/logreg_indication.joblib",
            "experiments/image_baseline/image_head_final.pth", "app/data/decision_rules.json", "app/pipelines/staged.py"} <= files
    assert "scripts/eval_test_split.py" not in files
    assert all(len(v) == 64 for v in s["files"].values()) and len(s["backbone_weights"]) == 64 and s["git_head"]


def test_the_hashes_follow_the_contents_of_each_frozen_file(fake_system):
    before = ev.frozen_system()
    (fake_system / "app/data/decision_rules.json").write_text("changed")
    after = ev.frozen_system()
    assert [f for f in before["files"] if before["files"][f] != after["files"][f]] == ["app/data/decision_rules.json"]
    ev.BACKBONE.write_text("other weights")
    assert ev.frozen_system()["backbone_weights"] != before["backbone_weights"]


# ------------------------------------------------------------------ one case through the prediction step
def fake_predictors(fail=None):
    def make(m):
        def f(x):
            if m == fail:
                raise RuntimeError("model broke")
            return {"probabilities": {"Normal": 0.2, "Pneumonia": 0.7, "Lung Cancer": 0.1}}
        return f
    return {m: make(m) for m in ev.MODS}


def row(tmp_path, indication="Fever and cough.", labs=True):
    rep = tmp_path / "r.txt"
    rep.write_text((f"INDICATION: {indication}\n\n" if indication else "") + "FINDINGS: clear.\n")      # no indication: no such section
    r = {"study_id": 7, "subject_id": 3, "image_path": "film.jpg", "report_path": str(rep)}
    from explain.blood import RAW_LABS
    r.update({c: (5.0 if labs else np.nan) for c in RAW_LABS})
    return pd.Series(r)


def test_a_case_gets_all_three_models_probabilities_and_the_apps_idea_of_which_inputs_it_has(tmp_path):
    rec = ev.predict_row(row(tmp_path), fake_predictors())
    assert rec["has_image"] and rec["has_blood"] and rec["has_text"] and rec["errors"] == ""
    assert rec["image_p_pneumonia"] == 0.7
    bare = ev.predict_row(row(tmp_path, indication="", labs=False), fake_predictors())
    assert bare["has_image"] and not bare["has_blood"] and not bare["has_text"]
    assert bare["blood_p_normal"] == 0.2                                                        # still scored, for the validation convention


def test_a_model_that_fails_on_a_case_is_recorded_and_treated_as_absent(tmp_path):
    rec = ev.predict_row(row(tmp_path), fake_predictors(fail="blood"))
    assert "blood: RuntimeError" in rec["errors"] and not rec["has_blood"] and np.isnan(rec["blood_p_normal"])
    assert rec["has_image"] and rec["has_text"]


# ------------------------------------------------------------------ the numbers the headline rests on, against independent computations
def test_the_headline_numbers_match_scikit_learn():
    from sklearn.metrics import f1_score, roc_auc_score
    rng = np.random.default_rng(6)
    y, P = fake_probs(rng, 500)
    mask = {m: np.ones(500, bool) for m in ev.MODS}
    rows = ev.per_row(P, mask, True, load_rules(), 0.8)
    s = ev.stats(np.arange(500), y, P, rows)
    pred = rows["p"].argmax(1)
    assert s["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert [s["f1_normal"], s["f1_pneumonia"], s["f1_cancer"]] == pytest.approx(list(f1_score(y, pred, average=None)))
    assert s["accuracy"] == pytest.approx((pred == y).mean())
    assert s["auc_cancer"] == pytest.approx(roc_auc_score(y == 2, rows["p"][:, 2]))
    assert s["image_macro_f1"] == pytest.approx(f1_score(y, P["image"].argmax(1), average="macro"))
    assert s["brier"] == pytest.approx(((rows["p"] - np.eye(3)[y]) ** 2).sum(1).mean())


def test_calibration_applies_the_temperature_of_the_inputs_the_case_had():
    from app.agent.decision import apply_temperature
    rng = np.random.default_rng(7)
    y, P = fake_probs(rng, 60)
    mask = {m: np.ones(60, bool) for m in ev.MODS}
    mask["blood"][:20] = False
    rules = load_rules()
    rows = ev.per_row(P, mask, False, rules, 0.8)
    t_all, t_two = rules["by_inputs"]["blood+image+text"]["temperature"], rules["by_inputs"]["image+text"]["temperature"]
    assert t_all != t_two
    assert np.allclose(rows["cal"][0], apply_temperature(rows["p"][0], t_two))
    assert np.allclose(rows["cal"][30], apply_temperature(rows["p"][30], t_all))
    assert not np.allclose(rows["cal"], rows["p"])
    s = ev.stats(np.arange(60), y, P, rows)
    assert s["ece_calibrated"] != s["ece"]


def test_the_deployed_system_and_the_validation_convention_differ_exactly_when_inputs_are_missing():
    rng = np.random.default_rng(8)
    y, P = fake_probs(rng, 300)
    full = {m: np.ones(300, bool) for m in ev.MODS}
    gaps = {m: v.copy() for m, v in full.items()}
    gaps["blood"][::2] = False
    gaps["text"][::5] = False
    same = ev.evaluate(P, full, y, np.arange(300), load_rules(), b=10)
    assert same["deployed"]["point"] == same["validation_convention"]["point"]
    differ = ev.evaluate(P, gaps, y, np.arange(300), load_rules(), b=10)
    assert differ["deployed"]["point"] != differ["validation_convention"]["point"]
    assert differ["validation_convention"]["point"] == same["validation_convention"]["point"]       # the convention ignores the mask
    assert differ["inputs_present"]["blood"] == pytest.approx(0.5)


def test_an_alert_that_the_rules_call_uninformative_is_not_counted():
    # with the image alone the rules found the cancer alert no better than chance, so it is not offered (MIN_ALERT_LIFT)
    rng = np.random.default_rng(9)
    y, P = fake_probs(rng, 300, skill=2.0)
    image_only = {"image": np.ones(300, bool), "blood": np.zeros(300, bool), "text": np.zeros(300, bool)}
    rows = ev.per_row(P, image_only, False, load_rules(), 0.8)
    assert not rows["informative"].any() and rows["flags"][0.6].any()
    s = ev.stats(np.arange(300), y, P, rows)
    assert s["alert_0.6_per_100"] == 0.0 and s["alert_0.6_recall"] == 0.0
    all_inputs = ev.per_row(P, {m: np.ones(300, bool) for m in ev.MODS}, False, load_rules(), 0.8)
    assert all_inputs["informative"].all()


def test_the_split_the_script_reads_is_the_sealed_one_unless_a_rehearsal_says_otherwise():
    assert ev.TEST_CSV.name == "test.csv" and ev.TEST_CSV.parent.name == "processed"
    assert ev.OUT.name == "test_evaluation" and ev.RESULTS.name == "results.json"


# ------------------------------------------------------------------ the whole procedure, rehearsed on validation studies
@pytest.mark.slow
def test_rehearsal_on_validation_studies_runs_both_steps_reproduces_stored_predictions_and_locks(tmp_path, monkeypatch):
    """Needs the models and the data. Runs the real --predict and --score on 120 validation studies in a scratch folder (the sealed
    split is not named), checks the live predictions equal the stored validation predictions, and that a second scoring is refused."""
    root = ev.ROOT
    val = pd.read_csv(root / "data" / "processed" / "val.csv").sample(120, random_state=1)
    sample = tmp_path / "rehearsal.csv"
    val.to_csv(sample, index=False)
    for name, value in (("OUT", tmp_path), ("TEST_CSV", sample), ("PREDICTIONS", tmp_path / "predictions.csv"),
                        ("RESULTS", tmp_path / "results.json"), ("FROZEN", tmp_path / "frozen_system.json"), ("B", 60)):
        monkeypatch.setattr(ev, name, value)
    ev.run_predict()
    pred = pd.read_csv(tmp_path / "predictions.csv").set_index("study_id")
    assert len(pred) == 120 and (pred["errors"].fillna("") == "").all()
    for m, (d, prefix) in {"image": ("image", "image_p_"), "blood": ("blood", "blood_p_"), "text": ("text", "text_indication_p_")}.items():
        stored = pd.read_csv(root / "experiments" / f"{d}_baseline" / "val_predictions.csv").set_index("study_id").loc[pred.index]
        live = pred[[f"{m}_p_{s}" for s in ev.SLUG]].values
        assert np.abs(live - stored[[prefix + s for s in ev.SLUG]].values).max() < 5e-4, m        # the prediction path equals the evaluated models
    ev.run_predict()                                                                         # resuming adds nothing
    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 120
    ev.run_score()
    r = json.loads((tmp_path / "results.json").read_text())
    assert r["test"]["n"] == 120 and r["validation_reference"]["n"] == 2315 and r["prediction_errors"] == 0
    assert 0.3 < r["test"]["validation_convention"]["point"]["macro_f1"] < 1.0
    with pytest.raises(SystemExit):
        ev.run_score()
