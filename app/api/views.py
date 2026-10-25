"""
How a stored case is shown to a client. One function builds the view, and both the plain GET and the live
stream use it, so they can never disagree about what a case looks like.

Two rules. Server file paths never leave the server: the uploaded image and the explanation figures are
shown as URLs under /api/cases/<id>/, and the database's own paths are not in the response. And the view is
read-only data: building it never changes a case.
"""
from __future__ import annotations

from app.database import repository as repo
from app.database.models import Case
from app.database.session import session_scope

TERMINAL = ("done", "partial", "failed")
FIGURE_NAMES = ("blood", "image", "text")


def figure_url(case_id: str, name: str) -> str:
    return f"/api/cases/{case_id}/figures/{name}.png"


def event_dict(e) -> dict:
    return {"id": e.id, "ts": e.ts.isoformat(), "stage": e.stage, "status": e.status, "seconds": e.seconds,
            "detail": e.detail}


def case_view(case: Case) -> dict:
    explanations = None
    if case.explanations:
        explanations = {m: {**e, "figure": figure_url(case.id, m) if e.get("figure") else None}
                        for m, e in case.explanations.items()}
    return {
        "id": case.id,
        "created_at": case.created_at.isoformat(),
        "updated_at": case.updated_at.isoformat(),
        "status": case.status,
        "stage": case.stage,
        "error": case.error,
        "inputs": {"image": bool(case.image_path),
                   "image_url": f"/api/cases/{case.id}/image" if case.image_path else None,
                   "labs": case.labs, "indication": case.indication},
        "prediction": case.prediction,
        "explanations": explanations,
        "evidence": case.evidence,
        "timings": case.timings or {},
        "events": [event_dict(e) for e in case.events],
    }


def case_summary(case: Case) -> dict:
    pred = (case.prediction or {}).get("belief", {})
    return {"id": case.id, "created_at": case.created_at.isoformat(), "status": case.status, "stage": case.stage,
            "predicted_class": pred.get("predicted_class"), "confidence": pred.get("confidence"),
            "modalities_used": (case.prediction or {}).get("modalities_used")}


def load_view(url: str | None, case_id: str) -> dict | None:
    with session_scope(url) as s:
        case = repo.get_case(s, case_id)
        return None if case is None else case_view(case)
