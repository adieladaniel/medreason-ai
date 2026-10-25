"""
The only place that reads or writes the tables. API handlers and the worker call these functions
and never build queries themselves, so the storage can change without touching either.

Every function takes an open Session and does not commit: the caller owns the transaction
(see session.session_scope). Anything stored as JSON goes through `jsonable` first, because the
pipeline's results contain sets and numpy numbers that the JSON column cannot serialise.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Case, Event, LlmCall, now

RESULT_FIELDS = {"prediction", "explanations", "evidence"}


def jsonable(obj: Any) -> Any:
    """Recursively turn pipeline output into plain JSON types: sets become sorted lists, numpy
    scalars and arrays become Python numbers and lists, non-string keys become strings."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj   # NaN and inf are not JSON
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((jsonable(v) for v in obj), key=str)
    if hasattr(obj, "tolist"):                       # numpy arrays and scalars
        return jsonable(obj.tolist())
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"cannot store {type(obj).__name__} as JSON")


def new_case_id() -> str:
    return uuid.uuid4().hex


def create_case(s: Session, *, indication: str | None = None, labs: dict | None = None,
                image_path: str | None = None, image_sha256: str | None = None, case_id: str | None = None) -> Case:
    """`case_id` lets the caller choose the id first, which the API needs: the uploaded image is saved in a
    folder named after the case before the row exists."""
    if not (indication or labs or image_path):
        raise ValueError("a case needs at least one of indication, labs, image")
    case = Case(id=case_id or new_case_id(), indication=indication or None, labs=jsonable(labs) if labs else None,
                image_path=image_path, image_sha256=image_sha256, status="queued", stage="queued", timings={})
    s.add(case)
    s.flush()
    return case


def get_case(s: Session, case_id: str) -> Case | None:
    return s.get(Case, case_id)


def delete_case(s: Session, case_id: str, case_dir: Path | None = None) -> bool:
    """Removes the case, its events and model calls, and its folder of uploaded and saved files. Returns
    whether there was such a case."""
    case = s.get(Case, case_id)
    if case is None:
        return False
    if case_dir is not None:
        shutil.rmtree(Path(case_dir) / case_id, ignore_errors=True)
    s.delete(case)
    return True


def list_cases(s: Session, limit: int = 50, offset: int = 0) -> list[Case]:
    return list(s.scalars(select(Case).order_by(Case.created_at.desc(), Case.id).limit(limit).offset(offset)))


def set_stage(s: Session, case: Case, stage: str, status: str = "running") -> None:
    case.stage, case.status = stage, status


def save_result(s: Session, case: Case, field: str, data: Any) -> None:
    if field not in RESULT_FIELDS:
        raise ValueError(f"unknown result field {field!r}")
    setattr(case, field, jsonable(data))


def record_timing(s: Session, case: Case, stage: str, seconds: float) -> None:
    case.timings = {**(case.timings or {}), stage: round(float(seconds), 3)}      # new dict: JSON columns do not track in-place edits


def add_event(s: Session, case: Case, stage: str, status: str, seconds: float | None = None,
              detail: dict | None = None) -> Event:
    ev = Event(case_id=case.id, stage=stage, status=status, seconds=seconds, detail=jsonable(detail) if detail else None)
    s.add(ev)
    s.flush()
    return ev


def events_after(s: Session, case_id: str, after_id: int = 0) -> list[Event]:
    """Events newer than `after_id`, oldest first. The progress stream polls with the last id it sent."""
    return list(s.scalars(select(Event).where(Event.case_id == case_id, Event.id > after_id).order_by(Event.id)))


def add_llm_call(s: Session, case: Case, *, model: str, purpose: str, prompt: str, response: str, seconds: float) -> LlmCall:
    call = LlmCall(case_id=case.id, model=model, purpose=purpose, prompt=prompt, response=response, seconds=round(seconds, 3))
    s.add(call)
    s.flush()
    return call


def fail_case(s: Session, case: Case, error: str) -> None:
    case.status, case.error = "failed", error[:2000]


def finish_case(s: Session, case: Case, error: str | None = None) -> None:
    """Mark the end of a run. With an error text the case is "partial": the prediction is stored but
    something after it failed, and the text says what."""
    case.stage = "done"
    case.status = "partial" if error else "done"
    case.error = error[:2000] if error else None


def interrupted_to_failed(s: Session) -> int:
    """At startup: a case still 'queued' or 'running' belongs to a process that no longer exists
    (the server was restarted mid-run), so it can never finish. Mark it failed so the client sees
    an answer and does not wait forever."""
    stuck = list(s.scalars(select(Case).where(Case.status.in_(("queued", "running")))))
    for c in stuck:
        c.status, c.error = "failed", "interrupted by a server restart"
        s.add(Event(case_id=c.id, stage=c.stage or "queued", status="failed", detail={"reason": "restart"}))
    return len(stuck)


def purge_older_than(s: Session, days: int, delete_files: bool = True, case_dir: Path | None = None) -> int:
    """Delete cases (events and LLM calls go with them) and their uploaded images past retention.
    `case_dir` is the folder that holds one sub-folder per case (the saved explanation figures); each
    purged case's sub-folder goes too."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    old = list(s.scalars(select(Case).where(Case.created_at < cutoff)))
    for c in old:
        if delete_files and c.image_path:
            Path(c.image_path).unlink(missing_ok=True)
        if delete_files and case_dir is not None:
            shutil.rmtree(Path(case_dir) / c.id, ignore_errors=True)
        s.delete(c)
    return len(old)
