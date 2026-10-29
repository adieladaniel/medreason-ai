"""
FastAPI app for MedReason-AI: the progress dashboard (static pages and the JSON behind them) and the case
API (app/api/cases.py) that runs a case through the pipeline in the background.

Run from the repo root:
    python -m uvicorn app.main:app --port 8000
then open http://localhost:8000 for the dashboard and http://localhost:8000/docs for the API.

create_app builds the app from explicit parts, so a test can give it a throwaway database, an upload folder
and a fake worker, and nothing global is shared. The module-level `app` is create_app() with the real
settings and the real models, which is what uvicorn serves.

At startup it creates the tables if they are missing, marks any case a previous process left unfinished as
failed (the worker does not survive a restart, so those can never finish), deletes cases past the retention
period together with their files, and starts the worker. With MEDREASON_WARM_UP_MODELS=true the worker loads
every model before it takes the first case.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import dashboard_data, health
from app.api import cases, examples, validation
from app.config import Settings, get_settings
from app.database import repository as repo
from app.database.session import init_db, session_scope
from app.observability import bind, configure_logging
from app.pipelines.runner import JobRunner

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
EXPLAIN_DIR = ROOT / "experiments" / "explain_samples"
log = logging.getLogger("medreason.api")

_CASE_PATH = re.compile(r"^/api/cases/([0-9a-f]{32})(?:/|$)")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestLogMiddleware:
    """One log line per request, tagged with a request id (taken from the caller's X-Request-ID header when it
    is a sane value, otherwise made up) and, for /api/cases/<id>/..., the case id. The id is sent back in the
    X-Request-ID header so a user can quote it.

    Written as plain ASGI and not BaseHTTPMiddleware, which buffers streaming bodies badly and would get in the
    way of the server-sent event stream. The logged time is the time to the first byte of the response, which for
    a stream is not its length.

    Level: reading things (GET with a good status) is DEBUG, because the Analyze page polls and a line per poll
    would bury everything else. Changes (POST, DELETE) and 4xx are INFO, 5xx is WARNING."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        sent = self._request_id(scope)
        match = _CASE_PATH.match(scope["path"])
        status = {"code": 500}

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", sent.encode("ascii")))
            await send(message)

        t0 = time.perf_counter()
        with bind(request_id=sent, case_id=match.group(1) if match else None):
            try:
                await self.app(scope, receive, send_with_id)
            except Exception:
                log.exception("%s %s raised", scope["method"], scope["path"])
                raise
            finally:
                code = status["code"]
                level = (logging.WARNING if code >= 500 else
                         logging.INFO if scope["method"] != "GET" or code >= 400 else logging.DEBUG)
                # A handler may learn the case id only after the middleware has bound its own (a POST creates it).
                case_id = (scope.get("state") or {}).get("case_id")
                with bind(case_id=case_id):
                    log.log(level, "%s %s -> %d in %.0f ms", scope["method"], scope["path"], code,
                            (time.perf_counter() - t0) * 1000, extra={"status_code": code})

    @staticmethod
    def _request_id(scope) -> str:
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                candidate = value.decode("latin-1")
                if _REQUEST_ID.match(candidate):
                    return candidate
        return uuid.uuid4().hex[:12]


def create_app(settings: Settings | None = None, runner: JobRunner | None = None, build_dashboard: bool = True,
               setup_logging: bool = False) -> FastAPI:
    """setup_logging installs the log handler (app/observability.py). It is off by default so tests that build
    many apps do not change global logging; the module-level `app` that uvicorn serves turns it on."""
    settings = settings or get_settings()
    if setup_logging:
        configure_logging(settings.log_level, settings.log_format)
    if runner is None and settings.run_worker:
        runner = JobRunner(settings.database_url, warm_up=settings.warm_up_models, figure_dir=settings.upload_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if build_dashboard and settings.rebuild_dashboard:
            dashboard_data.ensure_built()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        init_db(settings.database_url)
        with session_scope(settings.database_url) as s:
            interrupted = repo.interrupted_to_failed(s)
            purged = repo.purge_older_than(s, settings.retention_days, case_dir=settings.upload_dir)
        log.info("startup: %d interrupted case(s) marked failed, %d expired case(s) deleted", interrupted, purged,
                 extra={"interrupted": interrupted, "purged": purged})
        if app.state.runner is not None:
            app.state.runner.start()
        yield
        if app.state.runner is not None:
            app.state.runner.stop(timeout=5)

    app = FastAPI(title="MedReason-AI", version="0.2", lifespan=lifespan)
    app.state.settings = settings
    app.state.db_url = settings.database_url
    app.state.upload_dir = Path(settings.upload_dir)
    app.state.runner = runner
    app.state.poll_interval = 0.5          # seconds between looks at the database while streaming
    app.state.stream_timeout = 30 * 60     # a stream that has not finished by then is closed; the client can reconnect

    @app.exception_handler(validation.InputError)
    async def input_error(_request: Request, exc: validation.InputError):
        log.info("input refused: field=%s status=%d", exc.field, exc.status)     # the reason text can quote user input
        return JSONResponse({"detail": [{"field": exc.field, "message": exc.message}]}, status_code=exc.status)

    @app.get("/api/health")
    def liveness():
        """The process is up and answering. Does not look at anything else, so it never fails because a
        dependency is down; /api/ready is the one to ask whether a case can run."""
        return {"status": "ok"}

    @app.get("/api/ready")
    async def ready():
        """Probes the database, the worker, Qdrant and Ollama. 503 only when no case could run at all (database
        or worker), 200 with status "degraded" when only the evidence note would be missing."""
        status, checks = await run_in_threadpool(health.readiness, settings, app.state.runner)
        return JSONResponse({"status": status, "checks": checks}, status_code=503 if status == "down" else 200)

    @app.get("/api/stats")
    def stats():
        """Cases by status and seconds per stage (median, 95th percentile, worst) from what the database still
        holds, so the numbers cover the retention window."""
        runner_ = app.state.runner
        return health.usage_stats(app.state.db_url, runner_.pending() if runner_ else None)

    @app.get("/api/input-spec")
    def input_spec():
        """The fields, units and limits the Analyze form is built from."""
        return validation.input_spec(settings)

    @app.get("/api/dashboard")
    def dashboard():
        if not dashboard_data.OUT_PATH.exists():
            raise HTTPException(503, "dashboard data has not been built yet")
        return JSONResponse(json.loads(dashboard_data.OUT_PATH.read_text(encoding="utf-8")))

    @app.get("/api/gallery")
    def gallery():
        """Explainability sample cases, built from the files written by explain/run.py."""
        items = []
        for p in sorted(EXPLAIN_DIR.glob("explain_*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            sid = j["study_id"]
            mods = {}
            for m, r in j["modalities"].items():
                png = EXPLAIN_DIR / f"{m}_{sid}.png"
                mods[m] = {
                    "predicted_class": r["predicted_class"],
                    "probabilities": r["probabilities"],
                    "summary": r["summary"],
                    "method": r["attributions"]["method"],
                    "top": r["attributions"]["items"][:5],
                    "image": f"/samples/{m}_{sid}.png" if png.exists() else None,
                }
            items.append({"study_id": sid, "true_label": j["true_label"], "modalities": mods})
        return {"items": items}

    app.include_router(cases.router)
    app.include_router(examples.router)
    app.add_middleware(RequestLogMiddleware)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/samples", StaticFiles(directory=EXPLAIN_DIR, check_dir=False), name="samples")
    return app


app = create_app(setup_logging=True)
