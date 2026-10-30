"""Version 2, steps 3 and 4 (scripts/v2_evaluate.py): the adoption rule, the fusion of the two arms, the operating point and the guards that
keep the validation and held-out rows to one scoring. Synthetic arrays only; no data or model is needed."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("v2_evaluate", ROOT / "scripts" / "v2_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def passing():
    return {"auroc": {"value": 0.80, "low": 0.76, "high": 0.84}, "flagged_heldout": {"value": 0.40, "low": 0.3, "high": 0.5},
            "flagged_inscope": {"value": 0.10, "low": 0.08, "high": 0.12}, "macro_f1": {"value": 0.60, "low": 0.55, "high": 0.65},
            "cancer_flagged": {"value": 0.12, "low": 0.05, "high": 0.2}, "ece": {"value": 0.03, "low": 0.02, "high": 0.05}}


def test_a_result_that_meets_every_limit_is_adopted():
    ok, checks = ev.adopt(passing())
    assert ok and all(checks.values())


@pytest.mark.parametrize("stat, field, value, failed", [
    ("auroc", "value", 0.74, "auroc"),
    ("auroc", "low", 0.69, "auroc"),
    ("flagged_heldout", "value", 0.34, "flag_rates"),
    ("flagged_inscope", "value", 0.14, "flag_rates"),
    ("macro_f1", "value", 0.58, "three_class_macro_f1"),
    ("cancer_flagged", "value", 0.21, "cancers_flagged"),
    ("ece", "value", 0.06, "calibration"),
])
def test_each_limit_alone_stops_the_adoption(stat, field, value, failed):
    r = passing()
    r[stat][field] = value
    ok, checks = ev.adopt(r)
    assert not ok and [k for k, v in checks.items() if not v] == [failed]


def test_the_limits_are_the_ones_written_in_the_plan():
    plan = (ROOT / "experiments" / "v2_four_class" / "PLAN.md").read_text(encoding="utf-8").lower()
    for text in ("at least 0.75", "at least 0.70", "at least 35 percent", "at most 13 percent", "at least 0.583", "at most 20 percent", "at most 0.05"):
        assert text in plan, text
    assert ev.LIMITS == {"auroc": 0.75, "auroc_lower": 0.70, "flagged_heldout": 0.35, "flagged_inscope": 0.13, "macro_f1": 0.583,
                         "cancer_flagged": 0.20, "ece": 0.05}


def test_the_threshold_flags_the_requested_share():
    s = np.random.default_rng(0).random(10_000)
    tau = ev.threshold_for(s, 0.10)
    assert abs((s >= tau).mean() - 0.10) < 0.002


def test_temperature_scaling_recovers_one_for_calibrated_probabilities_and_never_changes_the_class():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2, 2, 2], 20_000)
    y = np.array([rng.choice(3, p=row) for row in p])
    assert abs(ev.temperature(p, y) - 1.0) < 0.15
    for t in (0.5, 1.0, 2.5):
        assert np.array_equal(ev.apply_temperature(p, t).argmax(1), p.argmax(1))


def synthetic_arms(monkeypatch):
    rng = np.random.default_rng(2)
    n = 50
    four = {m: rng.dirichlet([1, 1, 1, 1], n) for m in ev.lib.MODALITIES}
    two = {m: rng.dirichlet([1, 1], n) for m in ev.lib.MODALITIES}
    three = {m: rng.dirichlet([1, 1, 1], n) for m in ev.lib.MODALITIES}
    monkeypatch.setattr(ev, "load_arm", lambda arm, key: four if arm == "J" else two)
    monkeypatch.setattr(ev, "load_three_class", lambda key: three)
    return four, two, three


def test_arm_j_reads_the_fourth_probability_and_keeps_the_in_scope_class_among_the_first_three(monkeypatch):
    four, _, three = synthetic_arms(monkeypatch)
    f = ev.fuse("J", "oof")
    pooled = ev.lib.logpool(np.stack([four[m] for m in ev.lib.MODALITIES], axis=1))
    assert np.allclose(f["other"], pooled[:, 3])
    assert np.allclose(f["p3"].sum(1), 1) and np.array_equal(f["p3"].argmax(1), pooled[:, :3].argmax(1))
    assert np.allclose(f["v1_p3"], ev.lib.logpool(np.stack([three[m] for m in ev.lib.MODALITIES], axis=1)))


def test_arm_t_leaves_the_three_class_answer_exactly_as_version_1(monkeypatch):
    _, two, three = synthetic_arms(monkeypatch)
    f = ev.fuse("T", "val")
    assert np.array_equal(f["p3"], f["v1_p3"])
    assert np.allclose(f["other"], ev.lib.logpool(np.stack([two[m] for m in ev.lib.MODALITIES], axis=1))[:, 1])


def separated(n_neg=400, n_pos=400, gap=0.4, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n_neg)
    p3 = np.full((n_neg, 3), 0.1)
    p3[np.arange(n_neg), y] = 0.8
    neg = {"other": rng.random(n_neg) * 0.5, "p3": p3, "y": y}
    pos = {"other": rng.random(n_pos) * 0.5 + gap}
    return neg, pos, np.arange(n_neg) // 3, np.arange(n_pos) // 2


def test_statistics_and_the_bootstrap_on_a_separable_case():
    neg, pos, sn, sp = separated(gap=0.6)
    res = ev.with_intervals(neg, pos, tau=0.55, t=1.0, neg_subjects=sn, pos_subjects=sp, b=100)
    assert res["auroc"]["value"] > 0.99 and res["flagged_heldout"]["value"] > 0.9 and res["flagged_inscope"]["value"] < 0.15
    assert res["macro_f1"]["value"] == pytest.approx(1.0)
    for v in res.values():
        assert v["low"] <= v["value"] <= v["high"] or np.isnan(v["value"])


def test_the_bootstrap_is_reproducible_and_resamples_whole_patients():
    neg, pos, sn, sp = separated()
    a = ev.with_intervals(neg, pos, 0.4, 1.0, sn, sp, b=50)
    b = ev.with_intervals(neg, pos, 0.4, 1.0, sn, sp, b=50)
    assert a == b
    g = ev.groups_of(np.array([5, 5, 7, 7, 7, 9]))
    assert [len(x) for x in g] == [2, 3, 1]
    idx = ev.draw(g, np.random.default_rng(0))
    assert all(set(idx) >= set(x) or not set(idx) & set(x) for x in g)          # a patient comes in whole or not at all


def test_scoring_refuses_a_second_look(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ev, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="already scored once"):
        ev.score()


def test_scoring_refuses_without_a_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="--freeze"):
        ev.score()


def test_scoring_refuses_when_a_file_changed_after_the_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "OUT", tmp_path)
    (tmp_path / "frozen.json").write_text(json.dumps({"chosen_arm": "T", "prediction_files": {"a.npz": "old"}}), encoding="utf-8")
    monkeypatch.setattr(ev, "frozen_record", lambda: {"chosen_arm": "T", "prediction_files": {"a.npz": "new"}})
    with pytest.raises(SystemExit, match="changed after --freeze"):
        ev.score()


def test_the_scoring_script_names_no_test_table():
    code = (ROOT / "scripts" / "v2_evaluate.py").read_text(encoding="utf-8")
    assert "test.csv" not in code and "excluded_test" not in code
