"""
The case endpoints.

  POST   /api/cases                     submit a case (multipart form); answers 202 and the case id at once
  GET    /api/cases                     recent cases, newest first, with just the headline
  GET    /api/cases/{id}                the case as stored right now: any stage's result that exists
  GET    /api/cases/{id}/events         the audit trail: every stage start, finish and failure with seconds
  GET    /api/cases/{id}/stream         the same thing live (server-sent events) until the case is finished
  GET    /api/cases/{id}/llm-calls      every prompt sent to the language model and what it answered
  GET    /api/cases/{id}/figures/{name}.png    an explanation figure
  GET    /api/cases/{id}/image          the uploaded X-ray
  DELETE /api/cases/{id}                remove a finished case and its files

Why a job and not one long request: the pipeline takes between a second and a couple of minutes, depending
on whether the models are loaded. A request held open that long breaks on proxies and timeouts and cannot
survive a page reload. So POST stores the case, hands it to the worker and returns at once; the client reads
the case, or the stream, until its status is done, partial or failed.

No login. This is a local demonstration with no real patients, and the design notes say so. The cases still
expire (retention_days), and a case can be deleted.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api import validation
from app.api.views import FIGURE_NAMES, TERMINAL, case_summary, event_dict, iso_utc, load_view
from app.database import repository as repo
from app.database.session import session_scope
from app.observability import bind
from app.pipelines.staged import provided_inputs

log = logging.getLogger("medreason.api")
router = APIRouter(prefix="/api/cases", tags=["cases"])
CaseId = PathParam(pattern=r"^[0-9a-f]{32}$", description="a case id as returned by POST /api/cases")


def _links(case_id: str) -> dict:
    base = f"/api/cases/{case_id}"
    return {"self": base, "events": f"{base}/events", "stream": f"{base}/stream"}


@router.post("", status_code=202)
def submit_case(request: Request, image: UploadFile | None = File(None), indication: str | None = Form(None),
                labs: str | None = Form(None)):
    st = request.app.state
    cfg = st.settings
    if st.runner is None:
        raise HTTPException(503, "this server is running without a worker, so it cannot process cases")

    text = validation.clean_indication(indication, cfg.max_indication_chars)
    lab_values = validation.parse_labs(labs)

    data = b""
    if image is not None:
        limit = cfg.max_upload_mb * 1024 * 1024
        data = image.file.read(limit + 1)                  # one byte over the limit is enough to know it is too big
        if len(data) > limit:
            raise validation.InputError("image", f"the file is larger than {cfg.max_upload_mb} MB", status=413)
    info = validation.check_image(data, cfg.max_image_pixels, guard=cfg.xray_guard) if data else None

    if not provided_inputs(text, lab_values, "image" if data else None):
        raise validation.InputError("form", "give at least one of: a chest X-ray, lab values, or a clinical indication")
    if st.runner.pending() >= cfg.max_queue:
        raise HTTPException(429, f"{st.runner.pending()} cases are already waiting; try again in a few minutes",
                            headers={"Retry-After": "60"})

    case_id = repo.new_case_id()
    image_path = sha = None
    if data:
        folder = st.upload_dir / case_id
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / f"input.{info.extension}"
        image_path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
    try:
        with session_scope(st.db_url) as s:
            repo.create_case(s, case_id=case_id, indication=text, labs=lab_values,
                             image_path=str(image_path) if image_path else None, image_sha256=sha)
    except Exception:
        if data:
            for f in (st.upload_dir / case_id).glob("*"):
                f.unlink(missing_ok=True)
            (st.upload_dir / case_id).rmdir()
        raise
    ahead = st.runner.pending()                            # counted before this case joins the queue
    st.runner.submit(case_id)
    request.state.case_id = case_id                        # the request log line is written after this returns
    with bind(case_id=case_id):
        log.info("case submitted (image: %s, labs: %d, indication: %s, %d ahead in the queue)", "yes" if data else "no",
                 len(lab_values or {}), "yes" if text else "no", ahead)     # what was sent, never its content
    return JSONResponse({"id": case_id, "status": "queued", "links": _links(case_id)}, status_code=202,
                        headers={"Location": f"/api/cases/{case_id}"})


@router.get("")
def list_recent(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    with session_scope(request.app.state.db_url) as s:
        return {"cases": [case_summary(c) for c in repo.list_cases(s, limit, offset)]}


@router.get("/{case_id}")
def get_case(request: Request, case_id: str = CaseId):
    view = load_view(request.app.state.db_url, case_id)
    if view is None:
        raise HTTPException(404, "no such case")
    return view


@router.get("/{case_id}/events")
def get_events(request: Request, case_id: str = CaseId, after: int = Query(0, ge=0)):
    with session_scope(request.app.state.db_url) as s:
        if repo.get_case(s, case_id) is None:
            raise HTTPException(404, "no such case")
        return {"events": [event_dict(e) for e in repo.events_after(s, case_id, after)]}


@router.get("/{case_id}/llm-calls")
def get_llm_calls(request: Request, case_id: str = CaseId):
    with session_scope(request.app.state.db_url) as s:
        case = repo.get_case(s, case_id)
        if case is None:
            raise HTTPException(404, "no such case")
        return {"calls": [{"id": c.id, "ts": iso_utc(c.ts), "model": c.model, "purpose": c.purpose,
                           "prompt": c.prompt, "response": c.response, "seconds": c.seconds} for c in case.llm_calls]}


def _poll(url: str | None, case_id: str, cursor: int):
    """One look at the database for the stream: new events, the current view, and a signature that changes
    whenever anything a client would want to redraw has changed."""
    view = load_view(url, case_id)
    if view is None:
        return None
    new = [e for e in view["events"] if e["id"] > cursor]
    return new, view, (view["updated_at"], view["status"], view["stage"], len(view["events"]))


@router.get("/{case_id}/stream")
async def stream_case(request: Request, case_id: str = CaseId):
    """Server-sent events: `stage` for each new audit-trail event, `case` with the full view whenever it
    changed, and `done` once, when the case reaches done, partial or failed. A client that connects late
    gets everything that already happened first."""
    st = request.app.state
    if await run_in_threadpool(load_view, st.db_url, case_id) is None:
        raise HTTPException(404, "no such case")

    async def events():
        cursor, last_sig, started = 0, None, time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            snap = await run_in_threadpool(_poll, st.db_url, case_id, cursor)
            if snap is None:                                # deleted while watching
                yield {"event": "gone", "data": json.dumps({"id": case_id})}
                return
            new, view, sig = snap
            for e in new:
                cursor = e["id"]
                yield {"event": "stage", "id": str(e["id"]), "data": json.dumps(e)}
            if sig != last_sig:
                last_sig = sig
                yield {"event": "case", "data": json.dumps(view)}
            if view["status"] in TERMINAL:
                yield {"event": "done", "data": json.dumps({"status": view["status"], "error": view["error"]})}
                return
            if time.monotonic() - started > st.stream_timeout:
                yield {"event": "timeout", "data": json.dumps({"seconds": st.stream_timeout})}
                return
            await asyncio.sleep(st.poll_interval)

    return EventSourceResponse(events())


@router.get("/{case_id}/figures/{name}.png")
def get_figure(request: Request, case_id: str = CaseId, name: str = PathParam(pattern=r"^(blood|image|text)$")):
    assert name in FIGURE_NAMES
    path = request.app.state.upload_dir / case_id / f"{name}.png"
    if not path.is_file():
        raise HTTPException(404, "no such figure")
    return FileResponse(path, media_type="image/png")


@router.get("/{case_id}/image")
def get_input_image(request: Request, case_id: str = CaseId):
    st = request.app.state
    with session_scope(st.db_url) as s:
        case = repo.get_case(s, case_id)
        image_path = case.image_path if case else None
    if not image_path:
        raise HTTPException(404, "this case has no uploaded image")
    path = Path(image_path).resolve()
    if not path.is_file() or not path.is_relative_to(st.upload_dir.resolve()):      # never serve a path from outside the upload folder
        raise HTTPException(404, "this case has no uploaded image")
    return FileResponse(path, media_type="image/png" if path.suffix == ".png" else "image/jpeg")


@router.delete("/{case_id}", status_code=204)
def delete_case(request: Request, case_id: str = CaseId):
    st = request.app.state
    with session_scope(st.db_url) as s:
        case = repo.get_case(s, case_id)
        if case is None:
            raise HTTPException(404, "no such case")
        if case.status in ("queued", "running"):
            raise HTTPException(409, "the case is still being processed; delete it once it has finished")
        repo.delete_case(s, case_id, st.upload_dir)
