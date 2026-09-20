"""
Belief state: a probability distribution over Normal, Pneumonia, Lung Cancer, updated as
each tool result comes in.

The pooling rule is the exact one validated in Phase 3 (scripts/train_fusion.py,
logpool_rule): the geometric mean of the available modalities' probability vectors,
renormalized. That script found this parameter-free rule beat every trained combiner it
was tested against (val macro F1 0.613 vs 0.585 for the best trained stacker), so the
belief state reuses it rather than inventing a new update rule.

Recomputing the geometric mean from scratch on every update, rather than folding each new
result into a running product, is deliberate: it means the belief after any subset of
tools has run is identical to that subset's row in the Phase 3 ablation table
(experiments/fusion/metrics.json, ablation_val_macro_f1), so every number this agent
produces has already been measured, not derived from math that was never validated.
"""
from __future__ import annotations

import numpy as np

from app.agent import LABEL_NAMES


class BeliefState:
    def __init__(self):
        self.evidence: dict[str, np.ndarray] = {}
        self.history: list[dict] = []

    def update(self, modality: str, probabilities: dict) -> np.ndarray:
        """Add one modality's result and return the belief after including it."""
        self.evidence[modality] = np.array([probabilities[c] for c in LABEL_NAMES], dtype=float)
        current = self.pool()
        self.history.append({
            "modality": modality,
            "probabilities": dict(zip(LABEL_NAMES, self.evidence[modality].round(4).tolist())),
            "belief_after": dict(zip(LABEL_NAMES, current.round(4).tolist())),
            "predicted_class": LABEL_NAMES[int(current.argmax())],
        })
        return current

    def pool(self) -> np.ndarray:
        if not self.evidence:
            return np.full(3, 1 / 3)
        stacked = np.clip(np.stack(list(self.evidence.values())), 1e-6, 1.0)
        pooled = np.exp(np.log(stacked).mean(axis=0))
        return pooled / pooled.sum()

    def as_dict(self) -> dict:
        p = self.pool()
        return {
            "probabilities": dict(zip(LABEL_NAMES, p.round(4).tolist())),
            "predicted_class": LABEL_NAMES[int(p.argmax())],
            "confidence": round(float(p.max()), 4),
            "modalities_used": sorted(self.evidence.keys()),
            "history": self.history,
        }
