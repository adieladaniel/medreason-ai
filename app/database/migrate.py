"""
Bring a database to the current schema with Alembic (migrations/), the one way the tables are created and changed.

Three situations at startup, and what each does:

  empty database             run every migration, which creates the tables
  database with a version    run the migrations it has not had yet (usually none)
  database from before this  the tables exist (they were made by Base.metadata.create_all) but there is no version record. It is
                             checked against the baseline schema below and, if it matches, marked as being at the baseline, so
                             the migrations that come after the baseline apply to it. A database that does not match is refused
                             with a message, because marking an unknown schema as known would let later migrations fail halfway
                             or silently do the wrong thing.

Limits: one process migrates at a time. The compose stack has one API container, so two starting together against the same
Postgres is not handled (a lock around the upgrade would be the fix).
"""
from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from app.database.models import Base

log = logging.getLogger("medreason.database")

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "0001"
# The columns the baseline revision creates, by table. A pre-Alembic database is stamped only if it has exactly these. Names only:
# types are spelled differently by SQLite and Postgres. Written out here, not read from the models, because the models move on
# and this describes the database that existed at the baseline (tests/test_migrations.py checks it against the real baseline).
BASELINE_COLUMNS = {
    "cases": {"id", "created_at", "updated_at", "status", "stage", "error", "indication", "labs", "image_path", "image_sha256",
              "prediction", "explanations", "evidence", "timings"},
    "events": {"id", "case_id", "ts", "stage", "status", "seconds", "detail"},
    "llm_calls": {"id", "case_id", "ts", "model", "purpose", "prompt", "response", "seconds"},
}


class SchemaMismatch(RuntimeError):
    """The database is not in a state the migrations can take over."""


def config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))      # relative to the process otherwise
    cfg.attributes["url"] = url
    return cfg


def head_revision() -> str:
    heads = ScriptDirectory.from_config(config("unused")).get_heads()
    if len(heads) != 1:
        raise SchemaMismatch(f"the migrations have {len(heads)} heads ({', '.join(heads)}); merge them with `alembic merge`")
    return heads[0]


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def schema_differences(engine: Engine) -> list:
    """What `alembic revision --autogenerate` would want to change. Empty when the database matches the models."""
    with engine.connect() as conn:
        return compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)


def _check_pre_alembic(engine: Engine, tables: set[str]) -> None:
    missing = sorted(set(BASELINE_COLUMNS) - tables)
    if missing:
        raise SchemaMismatch(f"the database has some of the service's tables but not {', '.join(missing)}, and no migration "
                             "record, so it cannot be told apart from a broken one. Restore it or start from an empty database.")
    insp = inspect(engine)
    for table, expected in BASELINE_COLUMNS.items():
        have = {c["name"] for c in insp.get_columns(table)}
        if have != expected:
            raise SchemaMismatch(f"table {table} does not match the baseline schema (missing {sorted(expected - have) or 'none'}, "
                                 f"unexpected {sorted(have - expected) or 'none'}), and there is no migration record to say what "
                                 "else was changed. Fix it by hand or start from an empty database.")


def upgrade(engine: Engine, url: str) -> str:
    """Migrate to the newest revision and return it."""
    tables = set(inspect(engine).get_table_names())
    cfg = config(url)
    if "alembic_version" not in tables and tables & set(BASELINE_COLUMNS):
        _check_pre_alembic(engine, tables)
        command.stamp(cfg, BASELINE)
        log.info("database from before migrations matched the baseline schema and was marked as revision %s", BASELINE)
    before = current_revision(engine)
    command.upgrade(cfg, "head")
    after = current_revision(engine)
    gone = sorted(set(BASELINE_COLUMNS) - set(inspect(engine).get_table_names()))
    if gone:
        raise SchemaMismatch(f"the database says it is at revision {after} but the table(s) {', '.join(gone)} are missing "
                             "(dropped by hand?). Drop the alembic_version table too to start again from an empty schema.")
    if before != after:
        log.info("database migrated from %s to %s", before or "empty", after)
    return after
