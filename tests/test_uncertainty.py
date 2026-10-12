"""The helpers behind scripts/eval_uncertainty.py, checked on synthetic data where the right answer is known."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from scripts import eval_uncertainty as u

rng = np.random.default_rng(0)


def synthetic(n, sharpen=1.0):
    """Probabilities P and labels drawn FROM P (so P is calibrated), optionally made over-confident."""
    P = rng.dirichlet([2.0, 1.5, 0.7], n)
    y = np.array([rng.choice(3, p=p) for p in P])
    if sharpen != 1.0:
        Q = P ** sharpen
        P = Q / Q.sum(1, keepdims=True)
    return P, y


def test_f1_matches_sklearn_and_marks_absent_classes_nan():
    y = rng.integers(0, 3, 500)
    pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, 3, 500))
    assert np.allclose(u.f1s(y, pred), f1_score(y, pred, average=None, labels=[0, 1, 2]))
    only01 = np.array([0, 1, 0, 1, 1])
    f = u.f1s(only01, only01)
    assert np.isnan(f[2]) and np.nanmean(f) == 1.0       # a class with no cases is not scored as zero


def test_ece_is_small_for_calibrated_and_large_for_overconfident_probabilities():
    P, y = synthetic(20000)
    assert u.ece(P, y) < 0.03
    P2, y2 = synthetic(20000, sharpen=3.0)
    assert u.ece(P2, y2) > 0.10


def test_temperature_scaling_recovers_overconfidence_and_improves_log_loss():
    P, y = synthetic(20000, sharpen=3.0)
    T = u.fit_temperature(P, y)
    assert T > 1.5                                        # needs softening
    assert u.nll(u.apply_temperature(P, T), y) < u.nll(P, y)
    assert (u.apply_temperature(P, T).argmax(1) == P.argmax(1)).all()   # never changes the predicted class


def test_class_conditional_conformal_sets_reach_the_target_coverage_per_class():
    Pc, yc = synthetic(6000)
    Pt, yt = synthetic(6000)
    q = u.conformal_quantiles(Pc, yc, 0.1)
    sets = u.prediction_sets(Pt, q)
    for c in range(3):
        assert sets[yt == c, c].mean() == pytest.approx(0.9, abs=0.03)


def test_conformal_sets_grow_as_alpha_shrinks():
    P, y = synthetic(4000)
    small = u.prediction_sets(P, u.conformal_quantiles(P, y, 0.2)).sum(1).mean()
    large = u.prediction_sets(P, u.conformal_quantiles(P, y, 0.05)).sum(1).mean()
    assert large > small


def test_cluster_bootstrap_resamples_whole_subjects():
    subjects = np.repeat(np.arange(50), 4)                # 50 subjects, 4 studies each
    value = (subjects % 2).astype(float)                  # constant within a subject
    ci = u.cluster_bootstrap(lambda ix: {"m": value[ix].mean()}, subjects, b=300)
    lo, hi = ci["m"]
    assert lo < 0.5 < hi and hi - lo > 0.1                # wide, as with 50 clusters, not as with 200 independent rows


def test_pooled_probabilities_sum_to_one_and_match_the_belief_rule():
    from app.agent.belief import BeliefState
    X = rng.dirichlet([1, 1, 1], (5, 3)).reshape(5, 9)
    P = u.pooled(X)
    assert np.allclose(P.sum(1), 1)
    b = BeliefState()
    for i, m in enumerate(["image", "blood", "text"]):
        b.update(m, dict(zip(["Normal", "Pneumonia", "Lung Cancer"], X[0].reshape(3, 3)[i])))
    assert np.allclose(b.pool(), P[0])
