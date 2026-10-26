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

from pydantic import field_validator
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

    # How the page expresses uncertainty (rules in app/data/decision_rules.json, built by scripts/build_decision_rules.py).
    # prediction_set_coverage: 0.8 or 0.9. A higher value gives larger sets, "not sure between these" more often.
    # cancer_alert_recall: 0.5, 0.6, 0.7 or 0.8, the share of true cancers the alert is set to catch on training data.
    # Higher catches more cancers and raises more false alarms (validation, all inputs: 0.5 gives 8.9 alerts per 100 cases,
    # 0.6 gives 13.0, 0.7 gives 21.0, 0.8 gives 33.0). It is a product decision, so it is a setting.
    prediction_set_coverage: float = 0.8
    cancer_alert_recall: float = 0.6

    @field_validator("prediction_set_coverage")
    @classmethod
    def _coverage(cls, v: float) -> float:
        if v not in (0.8, 0.9):
            raise ValueError("prediction_set_coverage must be 0.8 or 0.9, the levels the rules file was built for")
        return v

    @field_validator("cancer_alert_recall")
    @classmethod
    def _recall(cls, v: float) -> float:
        if v not in (0.5, 0.6, 0.7, 0.8):
            raise ValueError("cancer_alert_recall must be 0.5, 0.6, 0.7 or 0.8, the levels the rules file was built for")
        return v

    # Cases hold clinical text, labs and an X-ray, so they are not kept forever.
    retention_days: int = 30

    # Logging (app/observability.py). "json" writes one object per line for a log collector; "text" is for a terminal.
    log_level: str = "INFO"
    log_format: str = "text"

    @field_validator("log_level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError("log_level must be DEBUG, INFO, WARNING or ERROR")
        return v

    @field_validator("log_format")
    @classmethod
    def _format(cls, v: str) -> str:
        if v not in ("text", "json"):
            raise ValueError("log_format must be text or json")
        return v

    # The local language model. Ollama runs on the Windows host (it already uses the GPU), so inside a
    # container the address is http://host.docker.internal:11434. Docker Desktop forwards that to the
    # host's localhost, so Ollama does not have to listen on the network.
    # Locally the default is the IPv4 address, not the name "localhost". Ollama listens on IPv4 only, Python tries
    # the IPv6 address first, and on this Windows laptop a refused loopback connection takes 2.0 s to be refused,
    # so "localhost" added about 2 s to every call (measured: 2.05 s against 0.01 to 0.03 s).
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:latest"

    # Evidence index. Unset means embedded mode: one folder per embedding model under qdrant_path, nothing
    # to start. A URL (for example http://qdrant:6333 in compose) means a Qdrant server, which has to be
    # filled once with `python -m app.rag.index --build` using the same setting.
    qdrant_url: str | None = None
    qdrant_path: Path = ROOT / "data" / "qdrant"


@lru_cache
def get_settings() -> Settings:
    return Settings()
