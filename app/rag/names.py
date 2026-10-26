"""
Where the evidence index lives, as pure functions of the settings.

These sit apart from app/rag/index.py because index.py imports torch and sentence-transformers, which
takes seconds and about a gigabyte. The readiness check (app/health.py) has to find the index without
loading any of that, so the naming rules are here and index.py imports them.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings

COLLECTION = "evidence"
DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"


def slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def collection_name(model_name: str, settings: Settings) -> str:
    """Embedded mode keeps one folder per model, so the collection name is the same in each. A server
    holds every model in one place, so there the model is part of the name and two models' vectors
    (which can have different sizes) never land in the same collection."""
    return f"{COLLECTION}_{slug(model_name)}" if settings.qdrant_url else COLLECTION


def embedded_index_file(model_name: str, settings: Settings) -> Path:
    """The file embedded Qdrant keeps the vectors in. Checking that it exists does not open the index,
    which matters because embedded Qdrant allows only one client per folder."""
    return settings.qdrant_path / slug(model_name) / "collection" / COLLECTION / "storage.sqlite"
