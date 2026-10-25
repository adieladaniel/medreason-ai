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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import dashboard_data
from app.api import cases, validation
from app.config import Settings, get_settings
from app.database import repository as repo
from app.database.session import init_db, session_scope
from app.pipelines.runner import JobRunner

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
EXPLAIN_DIR = ROOT / "experiments" / "explain_samples"
log = logging.getLogger("medreason.api")


def create_app(settings: Settings | None = None, runner: JobRunner | None = None, build_dashboard: bool = True) -> FastAPI:
    settings = settings or get_settings()
    if runner is None and settings.run_worker:
        runner = JobRunner(settings.database_url, warm_up=settings.warm_up_models, figure_dir=settings.upload_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if build_dashboard:
            dashboard_data.ensure_built()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        init_db(settings.database_url)
        with session_scope(settings.database_url) as s:
            interrupted = repo.interrupted_to_failed(s)
            purged = repo.purge_older_than(s, settings.retention_days, case_dir=settings.upload_dir)
        log.info("startup: %d interrupted case(s) marked failed, %d expired case(s) deleted", interrupted, purged)
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
        return JSONResponse({"detail": [{"field": exc.field, "message": exc.message}]}, status_code=exc.status)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

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

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/samples", StaticFiles(directory=EXPLAIN_DIR, check_dir=False), name="samples")
    return app


app = create_app()
