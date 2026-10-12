"""
Shared builders for the tests. The tests use synthetic agent results, never the saved real
cases in experiments/explain_samples/, because those hold MIMIC images and text and are
git-ignored, so a clean checkout (or CI) could not run tests that depended on them.
"""
import torch  # noqa: F401  (Windows: torch before sklearn/shap, see CLAUDE.md gotcha 7)

import pytest

LABELS = ["Normal", "Pneumonia", "Lung Cancer"]


def probs(normal, pneumonia, cancer):
    return {"Normal": normal, "Pneumonia": pneumonia, "Lung Cancer": cancer}


def tool(pred, p, items=None):
    return {"predicted_class": pred, "probabilities": p, "attributions": {"items": items or []}}


def item(feature, value, effect, direction="supports"):
    return {"feature": feature, "value": value, "effect": effect, "direction": direction}


def make_agent(pred="Pneumonia", final=None, status="agree", tools=None):
    final = final or {c: (0.8 if c == pred else 0.1) for c in LABELS}
    return {"belief": {"predicted_class": pred, "probabilities": final, "confidence": max(final.values())},
            "tool_outputs": tools or {},
            "consistency": {"status": status, "detail": "All three agree."}}


@pytest.fixture
def pneumonia_agent():
    """Pneumonia, all three modalities agree: WBC high (abnormal), albumin low (abnormal),
    monocytes normal, Consolidation and Lung Opacity present in the image."""
    p = probs(0.05, 0.9, 0.05)
    return make_agent("Pneumonia", final=p, status="agree", tools={
        "blood": tool("Pneumonia", p, [item("WBC", 31.0, 0.5), item("albumin", 2.8, 0.3),
                                        item("monocytes %", 5.0, 0.2), item("hemoglobin", 11.0, 0.1, "opposes")]),
        "image": tool("Pneumonia", p, [item("Lung Opacity", 0.8, 0.4), item("Consolidation", 0.7, 0.3),
                                        item("Edema", 0.2, 0.1), item("Fracture", 0.9, 0.5)]),
        "text": tool("Pneumonia", p, [item("fever", 1.0, 0.2), item("cough", 1.0, 0.1), item("the", 1.0, 0.3),
                                       item("pneumonia", 1.0, 0.4)]),
    })
