"""
The logic of the image-backbone experiment (scripts/eval_image_backbones.py) that does not need the extracted features: the
pre-registered adoption rule, the tie-break, the leakage readout, that its pooling is the app's pooling, and that neither script
can open the sealed test split. Written before the results were read, so the rule is tested as it was committed in
experiments/image_backbones/PLAN.md.
"""
from pathlib import Path

import numpy as np
import pytest

from app.agent.belief import BeliefState
from scripts import eval_image_backbones as ev
from scripts import extract_image_backbone_features as ex

ROOT = Path(__file__).resolve().parent.parent
GOOD = {"img_auc_cancer_lo": 0.56, "d_img_auc_cancer": (0.10, 0.03), "d_fused_f1": (0.01, -0.01), "d_fused_auc_cancer": (0.02,)}


def test_an_arm_that_meets_every_condition_is_adopted():
    assert ev.adopt(GOOD) == (True, [])


@pytest.mark.parametrize("change,fragment", [
    ({"d_img_auc_cancer": (0.04, 0.0)}, "not clearly above"),                   # paired interval touches 0
    ({"d_img_auc_cancer": (0.04, -0.02)}, "not clearly above"),
    ({"img_auc_cancer_lo": 0.50}, "still includes 0.50"),                       # better than control but not distinguishable from guessing
    ({"d_fused_f1": (-0.001, -0.01)}, "lower than the control"),                # fused macro F1 point estimate below the control
    ({"d_fused_f1": (0.0, -0.021)}, "more than 0.02 lower"),                    # may be meaningfully worse
    ({"d_fused_auc_cancer": (-0.001,)}, "fused Lung Cancer AUC is lower"),
])
def test_each_condition_alone_blocks_adoption(change, fragment):
    ok, why = ev.adopt({**GOOD, **change})
    assert not ok and any(fragment in w for w in why), why


def test_the_boundary_values_the_plan_allows_are_allowed():
    assert ev.adopt({**GOOD, "d_fused_f1": (0.0, -0.02), "d_fused_auc_cancer": (0.0,)})[0]


def test_the_highest_fused_macro_f1_wins_unless_two_are_within_0_005():
    assert ev.choose({}) is None
    assert ev.choose({"nih": {"fused_f1": 0.62}, "pc": {"fused_f1": 0.64}}) == "pc"
    # within 0.005: the arm that needs fewer backbones wins even if its score is slightly lower
    assert ev.choose({"nih": {"fused_f1": 0.640}, "nih+mimic": {"fused_f1": 0.644}}) == "nih"
    # more than 0.005 apart: the better score wins even though it needs two backbones
    assert ev.choose({"nih": {"fused_f1": 0.640}, "nih+mimic": {"fused_f1": 0.646}}) == "nih+mimic"


def test_the_leakage_readout_follows_the_plan_thresholds():
    assert "no evidence" in ev.leakage_readout(0.77, 0.75)
    assert "no evidence" in ev.leakage_readout(0.77, 0.80)                      # better is not inflation
    assert ev.leakage_readout(0.77, 0.73) == "inconclusive"
    assert "consistent with inflation" in ev.leakage_readout(0.77, 0.71)


def test_the_experiments_pooling_is_the_apps_pooling():
    rng = np.random.default_rng(0)
    p = [rng.dirichlet(np.ones(3), 25) for _ in range(3)]
    ours = ev.pool(*p)
    for i in range(25):
        b = BeliefState()
        for m, arr in zip(("image", "blood", "text"), p):
            b.update(m, dict(zip(("Normal", "Pneumonia", "Lung Cancer"), arr[i])))
        assert np.allclose(ours[i], b.pool(), atol=1e-9)


def test_the_head_learns_a_separable_problem():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 600)
    X = (np.eye(3)[y] * 3 + rng.normal(0, 0.5, (600, 3))).astype(np.float32)
    ev.torch.manual_seed(0)
    head = ev.train_head(X, y, epochs=60)            # 600 rows is 3 batches an epoch; the real run has 43, so give it more epochs
    assert (ev.predict(head, X).argmax(1) == y).mean() > 0.95


def test_the_stats_report_each_part_of_a_known_example():
    y = np.array([0, 1, 2, 2, 1, 0])
    perfect = np.eye(3)[y] * 0.9 + 0.05
    s = ev.stats(y, perfect, perfect, perfect)
    assert s["img_macro_f1"] == 1.0 and s["fused_macro_f1"] == 1.0 and s["img_auc_cancer"] == 1.0
    chance = np.full((6, 3), 1 / 3)
    assert ev.stats(y, chance, chance, chance)["img_auc_cancer"] == 0.5


def test_the_arms_are_the_ones_written_in_the_plan():
    plan = (ROOT / "experiments" / "image_backbones" / "PLAN.md").read_text(encoding="utf-8")
    assert set(ev.ARMS) == {"control", "nih", "pc", "nih+mimic"}
    for arm in ev.ARMS:
        assert f"| {arm} |" in plan, arm
    assert ev.ARMS["nih+mimic"] == ["nih", "mimic_ch"] and ev.ARMS["control"] == ["mimic_ch"]
    assert "all" not in ev.ARMS and not any("all" in b for b in sum(ev.ARMS.values(), []))     # the weights that include MIMIC


def test_neither_script_can_open_the_sealed_test_split():
    assert ex.SPLITS == ("val", "train")
    for name in ("eval_image_backbones.py", "extract_image_backbone_features.py"):
        code = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "test.csv" not in code and '"test"' not in code, name
