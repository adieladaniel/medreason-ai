"""
Missing-information planner: decides whether requesting another modality is worth it,
given what has run and how confident and consistent the belief is so far.

The two thresholds below (0.85 high confidence, 0.55 low confidence) are a deliberate
choice, not a measured one, there is no ground truth for "should have asked for more
data" to calibrate against. They implement the behavior described in the project's
architecture notes: skip requesting more input when confidence is already high, but flag
it when confidence is only middling. What is not a guess is which modality to recommend:
that uses MODALITY_PRIORITY (app/agent/tools.py), which is read from the real Phase 3
ablation (text contributes most, then blood, then image).
"""
from __future__ import annotations

import numpy as np

from app.agent.tools import MODALITY_PRIORITY

ALL_MODALITIES = {"image", "blood", "text"}
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.55

WHY = {
    "text": "the clinical indication was the strongest single modality for the rare Lung Cancer class in Phase 2 (F1 0.32 vs 0.17 for image and 0.20 for blood)",
    "blood": "labs add signal the other two miss, particularly for cases the image model calls confidently wrong",
    "image": "the chest X-ray is the primary diagnostic modality even though it contributed least to fusion overall",
}


def plan_missing_info(available: set[str], belief: np.ndarray, consistency: dict) -> dict:
    confidence = float(belief.max())
    missing = sorted(ALL_MODALITIES - available, key=MODALITY_PRIORITY.index)

    if not missing:
        if confidence >= LOW_CONFIDENCE:
            recommendation = None
        else:
            recommendation = (
                f"All three modalities ran and confidence is still only {confidence:.0%}. "
                "Consider clinical correlation or a follow-up study (for example CT) rather than "
                "another automated input; the system has nothing further to add."
            )
    elif confidence >= HIGH_CONFIDENCE and consistency["status"] == "agree":
        recommendation = None
    else:
        nxt = missing[0]
        recommendation = (
            f"Confidence is {confidence:.0%} using only {', '.join(sorted(available))}. "
            f"Recommend adding {nxt}: {WHY[nxt]}."
        )

    return {
        "confidence": round(confidence, 4),
        "missing_modalities": missing,
        "recommendation": recommendation,
    }
