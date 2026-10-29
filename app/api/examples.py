"""
Example cases for the Analyze page's "Load an example case" menu.

  GET /api/examples                 the list: id, title, note
  GET /api/examples/{id}            one example: its indication and labs, and whether it has an image
  GET /api/examples/{id}/image      the image

Two kinds. Curated cases are validation studies chosen by rule (scripts/build_demo_cases.py) and live in demo/cases/ next to a
manifest. That folder holds MIMIC images and clinical text, is git-ignored and is not part of any image, so these are served only
when the folder exists (mounted read-only by docker compose). The built-in ones are generated here and contain no patient data: a
picture that is not an X-ray, so the guard's refusal can be shown on any machine.

What is served is the title, the note for the presenter, the inputs and the image. The study id and the true label stay in the
manifest and are never sent, so the page cannot spoil the answer.
"""
from __future__ import annotations

import io
import json
import re

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image

router = APIRouter(prefix="/api/examples", tags=["examples"])
EXAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")

BUILTIN = {
    "not-an-xray-photo": {"title": "Not an X-ray: a colour picture", "kind": "built-in",
                          "note": "A colour image that is not a radiograph. The server should refuse it before any model runs."},
    "not-an-xray-noise": {"title": "Not an X-ray: random noise", "kind": "built-in",
                          "note": "Random grayscale noise. The server should refuse it before any model runs."},
}


def _builtin_image(example_id: str) -> bytes:
    rng = np.random.default_rng(7)
    if example_id == "not-an-xray-photo":
        h = w = 384
        a = np.stack([np.tile(np.linspace(30, 230, w), (h, 1)), np.tile(np.linspace(220, 40, h)[:, None], (1, w)),
                      np.full((h, w), 120.0)], axis=-1) + rng.normal(0, 6, (h, w, 3))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    else:
        img = Image.fromarray(rng.integers(0, 255, (384, 384), dtype=np.uint8), "L")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def curated(request: Request) -> dict[str, dict]:
    """The manifest's cases by id, or nothing when the demo folder is absent or unreadable."""
    folder = request.app.state.settings.demo_dir
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        return {c["id"]: c for c in manifest["cases"] if EXAMPLE_ID.match(c.get("id", "")) and (folder / c["id"] / "image.jpg").is_file()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


@router.get("")
def list_examples(request: Request):
    cases = curated(request)
    items = [{"id": i, "title": c["title"], "note": c["note"], "kind": "validation case"} for i, c in cases.items()]
    items += [{"id": i, "title": b["title"], "note": b["note"], "kind": b["kind"]} for i, b in BUILTIN.items()]
    return {"examples": items, "curated_available": bool(cases)}


def _find(request: Request, example_id: str):
    if not EXAMPLE_ID.match(example_id):
        raise HTTPException(404, "no such example")
    if example_id in BUILTIN:
        return "built-in", BUILTIN[example_id]
    case = curated(request).get(example_id)
    if case is None:
        raise HTTPException(404, "no such example")
    return "curated", case


@router.get("/{example_id}")
def get_example(request: Request, example_id: str):
    kind, c = _find(request, example_id)
    return {"id": example_id, "title": c["title"], "note": c["note"], "indication": c.get("indication", ""), "labs": c.get("labs", {}),
            "has_image": True, "kind": "validation case" if kind == "curated" else "built-in"}


@router.get("/{example_id}/image")
def get_example_image(request: Request, example_id: str):
    kind, _ = _find(request, example_id)
    if kind == "built-in":
        return Response(_builtin_image(example_id), media_type="image/png")
    folder = request.app.state.settings.demo_dir
    path = (folder / example_id / "image.jpg").resolve()
    if folder.resolve() not in path.parents:                       # defence in depth: the id was already checked against a pattern
        raise HTTPException(404, "no such example")
    return FileResponse(path, media_type="image/jpeg")
