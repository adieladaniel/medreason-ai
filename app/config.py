"""
Runtime settings, read from environment variables (prefix MEDREASON_) or a local .env file.

Why settings live in one place: the same code has to run on this laptop (SQLite file, embedded
Qdrant, Ollama on localhost) and inside docker compose (Postgres, a Qdrant server, an Ollama
container). Everything that differs between those is a setting here, so the difference is a set
of environment variables and not a code change.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDREASON_", env_file=".env", extra="ignore")

    # SQLite by default. In compose: postgresql+psycopg://user:password@postgres:5432/medreason
    database_url: str = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"
    upload_dir: Path = ROOT / "data" / "uploads"
    max_upload_mb: int = 10
    max_image_pixels: int = 50_000_000      # width times height; guards against a small file that unpacks to gigabytes
    max_indication_chars: int = 2000        # the text encoder reads about 512 tokens, so more is never used
    # Jobs waiting for the worker. One runs at a time, so a long queue only means a long wait; past this
    # the API answers 429 and the caller retries, instead of promising work it cannot start soon.
    max_queue: int = 10
    # Load every model when the server starts (about 1.2 GB of RAM, about 20 seconds once the files are cached) so the first case
    # does not pay for it. Off by default so a plain `uvicorn` start is quick and tests stay light.
    warm_up_models: bool = False
    run_worker: bool = True

    # Cases hold clinical text, labs and an X-ray, so they are not kept forever.
    retention_days: int = 30

    # The local language model. Ollama runs on the Windows host (it already uses the GPU), so inside a
    # container the address is http://host.docker.internal:11434. Docker Desktop forwards that to the
    # host's localhost, so Ollama does not have to listen on the network.
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"

    # Evidence index. Unset means embedded mode: one folder per embedding model under qdrant_path, nothing
    # to start. A URL (for example http://qdrant:6333 in compose) means a Qdrant server, which has to be
    # filled once with `python -m app.rag.index --build` using the same setting.
    qdrant_url: str | None = None
    qdrant_path: Path = ROOT / "data" / "qdrant"


@lru_cache
def get_settings() -> Settings:
    return Settings()
