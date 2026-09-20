"""
FastAPI app for the MedReason-AI progress dashboard.

Serves the single-page UI (app/static) and the JSON behind it. This is the first piece of
the Phase 7 backend: the same app will gain the pipeline endpoints (upload a case, run the
orchestrator, stream results) later, and the frontend is plain static files so it can be
hosted separately (for example on Vercel) and pointed at this API.

Run from the repo root:
    python -m uvicorn app.main:app --port 8000
then open http://localhost:8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import dashboard_data

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "app" / "static"
EXPLAIN_DIR = ROOT / "experiments" / "explain_samples"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    dashboard_data.ensure_built()
    yield


app = FastAPI(title="MedReason-AI", version="0.1", lifespan=lifespan)


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


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/samples", StaticFiles(directory=EXPLAIN_DIR, check_dir=False), name="samples")
