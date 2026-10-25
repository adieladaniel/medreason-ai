"""
How a stored case is shown to a client. One function builds the view, and both the plain GET and the live
stream use it, so they can never disagree about what a case looks like.

Two rules. Server file paths never leave the server: the uploaded image and the explanation figures are
shown as URLs under /api/cases/<id>/, and the database's own paths are not in the response. And the view is
read-only data: building it never changes a case.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.database import repository as repo
from app.database.models import Case
from app.database.session import session_scope

TERMINAL = ("done", "partial", "failed")
FIGURE_NAMES = ("blood", "image", "text")


def iso_utc(dt: datetime) -> str:
    """Times are stored in UTC, but SQLite hands them back without a zone, and a browser reads a time
    without a zone as local time (5.5 hours off on this machine). Always say it is UTC."""
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).isoformat()


def figure_url(case_id: str, name: str) -> str:
    return f"/api/cases/{case_id}/figures/{name}.png"


def event_dict(e) -> dict:
    return {"id": e.id, "ts": iso_utc(e.ts), "stage": e.stage, "status": e.status, "seconds": e.seconds,
            "detail": e.detail}


def case_view(case: Case) -> dict:
    explanations = None
    if case.explanations:
        explanations = {m: {**e, "figure": figure_url(case.id, m) if e.get("figure") else None}
                        for m, e in case.explanations.items()}
    return {
        "id": case.id,
        "created_at": iso_utc(case.created_at),
        "updated_at": iso_utc(case.updated_at),
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
    return {"id": case.id, "created_at": iso_utc(case.created_at), "status": case.status, "stage": case.stage,
            "predicted_class": pred.get("predicted_class"), "confidence": pred.get("confidence"),
            "modalities_used": (case.prediction or {}).get("modalities_used")}


def load_view(url: str | None, case_id: str) -> dict | None:
    with session_scope(url) as s:
        case = repo.get_case(s, case_id)
        return None if case is None else case_view(case)
