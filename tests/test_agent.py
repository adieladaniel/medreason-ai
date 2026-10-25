"""Belief state and consistency checker: the Phase 5 rules that the evaluation numbers rely on."""
import numpy as np
import pytest

from app.agent.belief import BeliefState
from app.agent.consistency import check_consistency
from tests.conftest import LABELS, probs


def test_empty_belief_is_uniform():
    assert np.allclose(BeliefState().pool(), 1 / 3)


def test_single_modality_belief_is_that_modality():
    b = BeliefState()
    b.update("blood", probs(0.2, 0.7, 0.1))
    assert np.allclose(b.pool(), [0.2, 0.7, 0.1])


def test_pool_is_normalised_geometric_mean():
    b = BeliefState()
    b.update("blood", probs(0.2, 0.7, 0.1))
    b.update("text", probs(0.1, 0.3, 0.6))
    g = np.sqrt(np.array([0.2, 0.7, 0.1]) * np.array([0.1, 0.3, 0.6]))
    assert np.allclose(b.pool(), g / g.sum())


def test_belief_does_not_depend_on_update_order():
    """The belief after a set of tools must match that set's Phase 3 ablation row, whatever order ran."""
    a, c = BeliefState(), BeliefState()
    for m, p in [("image", probs(0.5, 0.3, 0.2)), ("blood", probs(0.2, 0.5, 0.3)), ("text", probs(0.1, 0.2, 0.7))]:
        a.update(m, p)
    for m, p in [("text", probs(0.1, 0.2, 0.7)), ("image", probs(0.5, 0.3, 0.2)), ("blood", probs(0.2, 0.5, 0.3))]:
        c.update(m, p)
    assert np.allclose(a.pool(), c.pool())


def test_zero_probability_does_not_zero_the_belief():
    b = BeliefState()
    b.update("image", probs(0.0, 1.0, 0.0))
    b.update("blood", probs(0.0, 0.0, 1.0))
    assert np.isfinite(b.pool()).all() and b.pool().sum() == pytest.approx(1.0)


def _ev(*vectors):
    mods = ["image", "blood", "text"]
    return {mods[i]: np.array(v, dtype=float) for i, v in enumerate(vectors)}


N, P, C = [1, 0, 0], [0, 1, 0], [0, 0, 1]


def test_consistency_levels():
    assert check_consistency(_ev(P))["status"] == "single_modality"
    assert check_consistency(_ev(P, P))["status"] == "agree"
    assert check_consistency(_ev(P, C))["status"] == "conflict"
    assert check_consistency(_ev(P, P, P))["status"] == "agree"
    assert check_consistency(_ev(P, P, C))["status"] == "partial"
    assert check_consistency(_ev(N, P, C))["status"] == "conflict"


def test_partial_names_the_outlier():
    out = check_consistency(_ev(N, P, P))   # image is the odd one
    assert "Image is the outlier" in out["detail"]
    assert out["modality_predictions"]["image"] == LABELS[0]


# ------------------------------------------------------------------ the missing-information plan
def test_a_low_confidence_answer_with_every_input_is_called_uncertain_without_clinical_advice():
    """The project's scope excludes clinical advice and follow-up, and the page now shows this sentence."""
    from app.agent.planner import plan_missing_info
    belief = np.array([0.38, 0.40, 0.22])
    plan = plan_missing_info({"image", "blood", "text"}, belief, {"status": "conflict"})
    text = plan["recommendation"]
    assert "uncertain" in text and "no further input" in text
    import re
    assert not re.search(r"follow-up|CT|clinical|consider|should|recommend", text, re.I), text


def test_a_confident_answer_with_every_input_needs_no_note():
    from app.agent.planner import plan_missing_info
    assert plan_missing_info({"image", "blood", "text"}, np.array([0.05, 0.9, 0.05]), {"status": "agree"})["recommendation"] is None


def test_a_missing_input_is_recommended_in_the_measured_order():
    from app.agent.planner import plan_missing_info
    plan = plan_missing_info({"image"}, np.array([0.4, 0.4, 0.2]), {"status": "single_modality"})
    assert plan["missing_modalities"] == ["text", "blood"] and "adding text" in plan["recommendation"]
