"""
The agent's view of a case, from nothing but each modality's class probabilities.

This is the one place that turns model outputs into a belief, a consistency verdict and a
missing-information plan. app/agent/orchestrator.py uses it after running the explainers, and the
staged pipeline (app/pipelines/staged.py) uses it after the cheap predictions alone. It reads only the
probabilities and which modalities are present, so both paths give the same belief for the same models,
and it imports no model code, so it can be tested without loading anything.

Design rule kept: nothing here produces a class of its own. The predicted class is the argmax of the
pooled belief, which is the Phase 3 geometric mean of the models' probabilities.
"""
from __future__ import annotations

from app.agent import MODALITY_PRIORITY
from app.agent.belief import BeliefState
from app.agent.consistency import check_consistency
from app.agent.planner import plan_missing_info


def reason(probabilities: dict[str, dict]) -> dict:
    """Belief, consistency check and missing-information plan for the modalities present.

    The modalities are added in the validated priority order (text, blood, image) so the
    belief-after-each-step history always adds the most informative one first."""
    belief = BeliefState()
    for modality in MODALITY_PRIORITY:
        if modality in probabilities:
            belief.update(modality, probabilities[modality])
    pooled = belief.pool()
    consistency = check_consistency(belief.evidence)
    plan = plan_missing_info(set(belief.evidence.keys()), pooled, consistency)
    return {"belief": belief.as_dict(), "consistency": consistency, "missing_info_plan": plan}
