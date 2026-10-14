"""
Engine and session factory.

One engine per database URL. Sessions are short-lived and never shared between threads: the API
handlers and the background worker each open their own, which is what makes SQLite (one writer at
a time) behave under the worker thread. For SQLite the connection gets foreign keys switched on,
which it does not enforce by default, and write-ahead logging so a reader (the progress stream)
does not block the writer (the worker).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database.models import Base

_engines: dict[str, Engine] = {}
_factories: dict[str, sessionmaker] = {}


def make_engine(url: str) -> Engine:
    if url in _engines:
        return _engines[url]
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    _engines[url] = engine
    return engine


def session_factory(url: str | None = None) -> sessionmaker:
    url = url or get_settings().database_url
    if url not in _factories:
        _factories[url] = sessionmaker(make_engine(url), expire_on_commit=False)
    return _factories[url]


def init_db(url: str | None = None) -> None:
    """Create the tables if they are missing. Schema changes need a migration tool (Alembic) once
    real data exists, which is added with the container work."""
    url = url or get_settings().database_url
    if url.startswith("sqlite:///"):
        from pathlib import Path
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(make_engine(url))


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Commit on success, roll back on error, always close."""
    s = session_factory(url)()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
