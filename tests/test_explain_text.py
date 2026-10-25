"""The text explainer on indications too short for SHAP. The encoder is replaced by a lookup, so no model is needed."""
import numpy as np
import pytest

from explain import text as T

BASELINE = np.array([0.2, 0.5, 0.3])          # what the model says for an empty indication
TABLE = {"": BASELINE, "cough": np.array([0.7, 0.2, 0.1]), "[CLS]": BASELINE}


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(T, "predict_proba_texts", lambda texts: np.array([TABLE.get(t, BASELINE) for t in texts]))

    class NoShap:
        def __init__(self, *a, **k):
            pytest.fail("SHAP cannot run on fewer than two words and must not be called")
    monkeypatch.setattr(T.shap, "Explainer", NoShap)


@pytest.mark.parametrize("text", ["cough", "Cough.", " cough "])
def test_one_word_gets_its_exact_effect_instead_of_a_crash(fake_model, text):
    TABLE["Cough."] = TABLE["cough"]
    r = T.explain_text(text)
    assert r["predicted_class"] == "Normal" and r["probabilities"] == {"Normal": 0.7, "Pneumonia": 0.2, "Lung Cancer": 0.1}
    item, = r["attributions"]["items"]
    assert item["effect"] == pytest.approx(0.5) and item["direction"] == "supports" and "single-word" in r["attributions"]["method"]
    assert item["feature"].strip(".").lower() == "cough"


def test_a_word_that_lowers_the_prediction_is_marked_against(fake_model):
    TABLE["fever"] = np.array([0.1, 0.4, 0.5])        # predicts Lung Cancer, 0.5 against 0.3 for the empty indication
    r = T.explain_text("fever")
    assert r["predicted_class"] == "Lung Cancer" and r["attributions"]["items"][0]["effect"] == pytest.approx(0.2)
    TABLE["fever"] = np.array([0.6, 0.1, 0.3])
    r = T.explain_text("fever")
    assert r["predicted_class"] == "Normal" and r["attributions"]["items"][0]["effect"] == pytest.approx(0.4)


@pytest.mark.parametrize("text", ["", "   ", "???"])
def test_an_indication_with_no_words_has_nothing_to_attribute(fake_model, text):
    r = T.explain_text(text)
    assert r["attributions"]["items"] == [] and "no words" in r["summary"] and r["_raw"]["tokens"] == []


@pytest.mark.parametrize("text", ["cough", ""])
def test_the_figure_still_draws_for_short_indications(fake_model, tmp_path, text):
    out = T.plot_text(T.explain_text(text), tmp_path / "t.png")
    assert out.exists() and out.stat().st_size > 0
