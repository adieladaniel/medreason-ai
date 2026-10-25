"""
Uniform wrappers around the three Phase 4 explainers. Each explain.* module already
returns {predicted_class, probabilities, attributions, summary}, which is exactly what a
tool call needs, so there is nothing to reimplement here. This module exists only to give
the orchestrator one calling convention (a single positional argument, a shared name) in
place of three different keyword names (image_path, labs, indication_text).
"""
from __future__ import annotations

# Windows: import torch before sklearn/shap, or torch's DLL init can fail (WinError
# 1114). explain.blood and explain.text both need this; do it once here.
import torch  # noqa: F401

from explain import blood as _blood
from explain import image as _image
from explain import text as _text


def run_image(image_path: str) -> dict:
    return _image.explain_image(image_path)


def run_blood(labs: dict) -> dict:
    return _blood.explain_blood(labs)


def run_text(indication: str) -> dict:
    return _text.explain_text(indication)


TOOLS = {"image": run_image, "blood": run_blood, "text": run_text}

# The same three models without the attribution step: probabilities and nothing else. These are what
# makes the staged pipeline fast, because the slow part of a tool call is the explanation, not the model.
PREDICTORS = {"image": _image.predict_image, "blood": _blood.predict_blood, "text": _text.predict_text}

# MODALITY_PRIORITY (text, blood, image: the measured ablation order) is defined in app/agent/__init__.py
# and re-exported here so existing imports keep working.
from app.agent import MODALITY_PRIORITY  # noqa: E402,F401
