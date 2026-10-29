"""
The out-of-scope detector (app/agent/scope.py, scripts/train_out_of_scope.py): the feature definition shared by training and serving, the
pre-registered adoption rule, patient-grouped cross-validation, the judging arithmetic, and the promises about data (no test patients) and about
behaviour (the check never changes a prediction).
"""
from pathlib import Path

import numpy as np
import pytest

from app.agent import scope
from app.agent.belief import BeliefState
from scripts import train_out_of_scope as tos

ROOT = Path(__file__).resolve().parent.parent
CLS = ["Normal", "Pneumonia", "Lung Cancer"]


def draw(rng, n, with_blood=True, with_text=True):
    probs = lambda: rng.dirichlet(np.ones(3), n)          # noqa: E731
    return dict(xrv18=rng.random((n, 18)), image_p=probs(), blood_p=probs(), text_p=probs(),
                has_blood=np.full(n, with_blood), has_text=np.full(n, with_text))


# ------------------------------------------------------------------ the shared feature definition
@pytest.mark.parametrize("backbone,expected", [(False, 13), (True, 24)])
def test_the_feature_matrix_and_its_names_agree(backbone, expected):
    d = draw(np.random.default_rng(0), 7)
    X = scope.features(**d, with_backbone=backbone)
    assert X.shape == (7, expected) and len(scope.feature_names(backbone)) == expected and np.isfinite(X).all()


def test_only_the_eleven_informative_backbone_outputs_enter_and_the_constant_slots_do_not():
    assert len(scope.INFORMATIVE18) == 11 and not set(scope.INFORMATIVE18) & {2, 5, 6, 9, 11, 12, 13}
    d = draw(np.random.default_rng(1), 5)
    changed = dict(d)
    changed["xrv18"] = d["xrv18"].copy()
    changed["xrv18"][:, [2, 5, 6, 9, 11, 12, 13]] = 99.0              # the untrained slots
    assert np.array_equal(scope.features(**d), scope.features(**changed))
    changed["xrv18"][:, 4] += 0.5                                      # a real output
    assert not np.array_equal(scope.features(**d), scope.features(**changed))


def test_the_pooled_summary_is_the_apps_pooling_over_the_inputs_the_case_has():
    rng = np.random.default_rng(2)
    d = draw(rng, 30)
    d["has_blood"] = rng.random(30) > 0.5
    d["has_text"] = rng.random(30) > 0.5
    pooled = scope.pooled(d["image_p"], d["blood_p"], d["text_p"], d["has_blood"], d["has_text"])
    for i in range(30):
        b = BeliefState()
        b.update("image", dict(zip(CLS, d["image_p"][i])))
        if d["has_blood"][i]:
            b.update("blood", dict(zip(CLS, d["blood_p"][i])))
        if d["has_text"][i]:
            b.update("text", dict(zip(CLS, d["text_p"][i])))
        assert np.allclose(pooled[i], b.pool(), atol=1e-9)


def test_features_for_one_case_equal_the_same_case_inside_a_batch():
    d = draw(np.random.default_rng(3), 6)
    batch = scope.features(**d)
    one = scope.features(d["xrv18"][2], d["image_p"][2], d["blood_p"][2], d["text_p"][2], [True], [True])
    assert np.allclose(batch[2], one[0])


# ------------------------------------------------------------------ serving: a gate, never a diagnosis
class FakeModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]] * len(x))


RULES = {"with_backbone": True, "threshold": 0.4, "operating_point": {"flag_rate_in_scope": 0.1, "catch_rate_held_out": 0.5}}


def test_a_case_above_the_threshold_is_flagged_and_below_is_not():
    d = draw(np.random.default_rng(4), 1)
    args = (d["xrv18"][0], d["image_p"][0], d["blood_p"][0], d["text_p"][0], True, True)
    assert scope.assess(*args, rules=RULES, model=FakeModel(0.55))["out_of_scope"] is True
    r = scope.assess(*args, rules=RULES, model=FakeModel(0.25))
    assert r["out_of_scope"] is False and r["score"] == 0.25 and r["threshold"] == 0.4
    assert r["flag_rate_in_scope"] == 0.1 and r["catch_rate_other"] == 0.5


def test_without_a_fitted_detector_the_check_says_nothing(monkeypatch):
    d = draw(np.random.default_rng(5), 1)
    args = (d["xrv18"][0], d["image_p"][0], d["blood_p"][0], d["text_p"][0], True, True)
    assert scope.assess(*args, rules={}, model=FakeModel(0.9)) is None                  # no rules
    monkeypatch.setattr(scope, "load_rules", lambda path=None: RULES)
    monkeypatch.setattr(scope, "_model", lambda: None)
    assert scope.assess(*args) is None                                                  # rules but no fitted model
    monkeypatch.setattr(scope, "load_rules", lambda path=None: None)
    assert scope.assess(*args, model=FakeModel(0.9)) is None                            # a model but no rules


def test_the_check_never_returns_or_changes_a_class():
    d = draw(np.random.default_rng(6), 1)
    before = {k: np.array(v, copy=True) for k, v in d.items()}
    r = scope.assess(d["xrv18"][0], d["image_p"][0], d["blood_p"][0], d["text_p"][0], True, True, rules=RULES, model=FakeModel(0.9))
    assert not any("class" in k or "label" in k or "diagnos" in k for k in r)
    for k in d:
        assert np.array_equal(d[k], before[k])
    source = (ROOT / "app" / "agent" / "scope.py").read_text(encoding="utf-8")
    assert "predicted_class" not in source.split("def assess")[1]


# ------------------------------------------------------------------ the pre-registered rule
GOOD = {"auroc": 0.80, "auroc_lo": 0.76, "catch_rate": 0.45, "false_flag_rate": 0.10, "f1_unflagged": 0.64, "f1_all": 0.613, "cancers_flagged": 0.12}


def test_a_detector_that_meets_every_condition_is_adopted():
    assert tos.adopt(GOOD) == (True, [])


@pytest.mark.parametrize("change,fragment", [
    ({"auroc": 0.74}, "AUROC"), ({"auroc_lo": 0.69}, "AUROC"), ({"catch_rate": 0.34}, "fewer than 35"),
    ({"false_flag_rate": 0.131}, "more than 13"), ({"f1_unflagged": 0.60}, "below the macro F1"), ({"cancers_flagged": 0.21}, "20 percent"),
])
def test_each_condition_alone_blocks_adoption(change, fragment):
    ok, why = tos.adopt({**GOOD, **change})
    assert not ok and any(fragment in w for w in why), why


def test_the_boundaries_the_plan_allows_are_allowed():
    assert tos.adopt({**GOOD, "auroc": 0.75, "auroc_lo": 0.70, "catch_rate": 0.35, "false_flag_rate": 0.13, "f1_unflagged": 0.613,
                      "cancers_flagged": 0.20})[0]


def test_the_simpler_arm_wins_a_near_tie_and_a_clearly_better_one_wins():
    assert tos.choose_arm({"A": 0.700, "B": 0.703, "C": 0.704}) == "A"
    assert tos.choose_arm({"A": 0.70, "B": 0.78, "C": 0.784}) == "B"
    assert tos.choose_arm({"A": 0.70, "B": 0.78, "C": 0.80}) == "C"
    assert list(tos.ARMS) == ["A", "B", "C"] and tos.ARMS["A"]["with_backbone"] is False and tos.ARMS["C"]["model"] == "hgb"


# ------------------------------------------------------------------ cross-validation holds out whole patients
def test_cross_validation_never_trains_on_a_patient_it_is_scoring():
    # every patient is wholly one class and the only feature identifies the patient: any leak of a patient across folds would make this
    # memorisable (AUROC near 1), while holding whole patients out leaves nothing to learn from (AUROC near 0.5)
    rng = np.random.default_rng(7)
    patients = np.repeat(np.arange(200), 3)
    label = rng.integers(0, 2, 200)[patients].astype(float)
    X = np.eye(200)[patients]
    from sklearn.metrics import roc_auc_score
    oof = tos.cross_validate(X, label, patients, "logreg")
    assert 0.35 < roc_auc_score(label, oof) < 0.65


def test_cross_validation_learns_a_real_signal():
    rng = np.random.default_rng(8)
    patients = np.repeat(np.arange(300), 2)
    y = rng.integers(0, 2, 300)[patients].astype(float)
    X = (y[:, None] * 1.5 + rng.normal(0, 1, (600, 4)))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, tos.cross_validate(X, y, patients, "logreg")) > 0.85


# ------------------------------------------------------------------ judging
def test_judging_a_perfect_detector_and_an_uninformative_one():
    n_in, n_ex = 200, 100
    in_true = np.r_[np.zeros(150, int), np.ones(40, int), np.full(10, 2)]
    in_pred = in_true.copy()
    s_in, s_ex = np.full(n_in, 0.1), np.full(n_ex, 0.9)
    s_in[:20] = 0.1
    subj_in, subj_ex = np.arange(n_in), 1000 + np.arange(n_ex)
    conf_in, conf_ex = np.full(n_in, 0.7), np.full(n_ex, 0.7)
    perfect = tos.judge(s_in, s_ex, 0.5, subj_in, subj_ex, in_pred, in_true, conf_in, conf_ex, b=30)
    p = perfect["point"]
    assert p["auroc"] == 1.0 and p["catch_rate"] == 1.0 and p["false_flag_rate"] == 0.0 and p["f1_unflagged"] == p["f1_all"] == 1.0
    assert p["auroc_existing_confidence"] == 0.5 and p["cancers_flagged"] == 0.0
    rng = np.random.default_rng(9)
    noisy = tos.judge(rng.random(n_in), rng.random(n_ex), 0.9, subj_in, subj_ex, in_pred, in_true, conf_in, conf_ex, b=30)
    assert 0.35 < noisy["point"]["auroc"] < 0.65 and 0.0 <= noisy["point"]["false_flag_rate"] <= 0.25


def test_flagging_removes_the_wrong_cases_raises_the_f1_of_what_is_kept_and_cancers_flagged_counts_only_cancers():
    in_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    in_pred = np.array([0, 0, 1, 0, 2, 0, 2, 2])             # two wrong: index 3 and 5
    s_in = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1])   # flags the two wrong ones and one correct Normal
    j = tos.judge(s_in, np.full(4, 0.95), 0.5, np.arange(8), 100 + np.arange(4), in_pred, in_true, np.full(8, 0.6), np.full(4, 0.6), b=20)["point"]
    assert j["false_flag_rate"] == 0.375 and j["f1_unflagged"] == 1.0 and j["f1_all"] < 1.0
    assert j["cancers_flagged"] == pytest.approx(1 / 4)        # one of the four true cancers, not 3 of 8 studies


def test_the_intervals_resample_patients_and_are_repeatable():
    rng = np.random.default_rng(10)
    n = 120
    args = (rng.random(n), rng.random(80) + 0.2, 0.7, np.repeat(np.arange(40), 3), np.repeat(np.arange(40), 2) + 500,
            rng.integers(0, 3, n), rng.integers(0, 3, n), rng.random(n), rng.random(80))
    a, b = tos.judge(*args, b=60), tos.judge(*args, b=60)
    assert a == b
    lo, hi = a["interval"]["auroc"]
    assert lo <= a["point"]["auroc"] <= hi


# ------------------------------------------------------------------ the data promises
def test_the_sealed_split_cannot_be_named_by_either_script_or_the_builder():
    assert tos.__name__ and "test" not in tos.ARMS
    for name in ("train_out_of_scope.py", "extract_scope_features.py"):
        code = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "test.csv" not in code and 'role="test"' not in code and "'test'" not in code, name
    from scripts import extract_scope_features as ext
    assert ext.ROLES == ("train", "val", "none")
    builder = (ROOT / "scripts" / "build_final_dataset.py").read_text(encoding="utf-8")
    assert 'choices=["unseen", "train", "val", "none"]' in builder and "args.role == \"test\"" not in builder


def test_the_detector_files_are_kept_out_of_git_and_images_but_its_small_rules_file_may_be_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "experiments/out_of_scope/feats_*.npz" in ignored and "*.joblib" in ignored
    assert "out_of_scope_rules" not in ignored                          # aggregate thresholds only, no identifiers


def test_the_clean_diagnostic_reports_every_arm_and_finds_a_planted_signal():
    rng = np.random.default_rng(11)

    def block(n_patients, per, shift, first_patient):
        n = n_patients * per
        d = draw(rng, n)
        d["xrv18"][:, 4] += shift                                  # a backbone output that differs between the groups
        d["subject_id"] = np.repeat(np.arange(first_patient, first_patient + n_patients), per)
        d["study_id"] = np.arange(n) + first_patient * 100
        return d
    d = tos.clean_diagnostic(block(150, 2, 0.0, 0), block(100, 2, 0.8, 1000), block(80, 2, 0.8, 2000))
    assert set(d["cv_auroc"]) == {"A", "B", "C"} and d["in_scope_validation"] == 300 and d["excluded_validation_patients"] == 200
    assert d["patients"] == 250
    assert d["cv_auroc"]["A"] < 0.62 and d["cv_auroc"]["B"] > 0.8 and d["cv_auroc"]["C"] > 0.8        # only the backbone arms can see it
    assert d["in_scope_oof_vs_no_split_excluded_auroc"]["B"] > 0.8


def test_no_unvalidated_detector_ships_so_the_check_is_dormant():
    # experiments/out_of_scope/RESULTS.md: the detector did not meet its pre-registered adoption rule, so nothing is written for the app to read.
    # Adopting a future detector means changing this test, the docs and the page's wording together, on purpose.
    assert not (ROOT / "app" / "data" / "out_of_scope_rules.json").exists()
    assert not (ROOT / "experiments" / "out_of_scope" / "detector.joblib").exists()
    scope.load_rules.cache_clear()
    assert scope.available() is False


def test_the_check_needs_both_the_rules_and_the_fitted_model(monkeypatch):
    monkeypatch.setattr(scope, "load_rules", lambda path=None: RULES)
    monkeypatch.setattr(scope, "_model", lambda: None)
    assert scope.available() is False                       # rules without a model
    monkeypatch.setattr(scope, "load_rules", lambda path=None: None)
    monkeypatch.setattr(scope, "_model", lambda: FakeModel(0.5))
    assert scope.available() is False                       # a model without rules
    monkeypatch.setattr(scope, "load_rules", lambda path=None: RULES)
    assert scope.available() is True


def test_the_clean_diagnostic_holds_whole_patients_out_so_a_memorised_patient_is_no_help():
    # no real difference between the groups, but every patient has a constant feature value that identifies them: memorisable if a patient's
    # studies were split across folds, useless if whole patients are held out together
    rng = np.random.default_rng(12)

    def block(n_patients, first):
        per, n = 8, n_patients * 8
        d = draw(rng, n)
        for col in (0, 1, 3, 4, 7, 8):                              # the identity is written into several backbone outputs
            d["xrv18"][:, col] = np.repeat(rng.random(n_patients) * 5, per)
        d["subject_id"] = np.repeat(np.arange(first, first + n_patients), per)
        d["study_id"] = np.arange(n) + first * 100
        return d
    d = tos.clean_diagnostic(block(120, 0), block(80, 1000), block(40, 2000))
    assert d["cv_auroc"]["B"] < 0.62 and d["cv_auroc"]["C"] < 0.62          # an ungrouped split scores about 0.57 and 1.00 on this data
