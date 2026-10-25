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
