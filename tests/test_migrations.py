"""
Schema migrations (Alembic, migrations/). The properties that matter: a model change cannot ship without a migration, an old
database is taken over safely and an unknown one is refused, and the baseline renders correctly for Postgres without needing one.
The Postgres run of the same checks is slow-marked (set MEDREASON_TEST_POSTGRES_URL, see tests/test_database.py).
"""
import ast
import io
import os
import re
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import inspect

from app.database import repository as repo
from app.database.migrate import (BASELINE, BASELINE_COLUMNS, ROOT, SchemaMismatch, config, current_revision, head_revision,
                                  schema_differences)
from app.database.models import Base
from app.database.session import init_db, make_engine, session_scope

POSTGRES_URL = os.environ.get("MEDREASON_TEST_POSTGRES_URL")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=[
    pytest.mark.slow, pytest.mark.skipif(not POSTGRES_URL, reason="set MEDREASON_TEST_POSTGRES_URL to run on Postgres")])])
def empty(request, tmp_path):
    """A URL whose database has no tables."""
    if request.param == "sqlite":
        yield f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
        return
    engine = make_engine(POSTGRES_URL)
    wipe(engine)
    yield POSTGRES_URL
    wipe(engine)
    engine.dispose()


def wipe(engine):
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")


# ------------------------------------------------------------------ the schema the migrations build
def test_migrating_an_empty_database_builds_exactly_what_the_models_describe(empty):
    init_db(empty)
    engine = make_engine(empty)
    assert current_revision(engine) == head_revision()
    assert set(BASELINE_COLUMNS) <= set(inspect(engine).get_table_names())
    assert schema_differences(engine) == [], "the models and the migrations disagree: add a migration (alembic revision --autogenerate)"


def test_a_model_change_without_a_migration_is_noticed(empty):
    init_db(empty)
    engine = make_engine(empty)
    with engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE cases ADD COLUMN sneaked_in TEXT")
    assert schema_differences(engine), "a column the models do not have went unnoticed"


def test_the_migrations_have_one_head_and_the_baseline_columns_are_what_the_baseline_creates(empty):
    assert head_revision()
    init_db(empty)
    insp = inspect(make_engine(empty))
    assert {t: {c["name"] for c in insp.get_columns(t)} for t in BASELINE_COLUMNS} == BASELINE_COLUMNS


def test_running_the_migrations_again_changes_nothing_and_keeps_the_data(empty):
    init_db(empty)
    with session_scope(empty) as s:
        cid = repo.create_case(s, indication="keep me").id
    init_db(empty)
    init_db(empty)
    with session_scope(empty) as s:
        assert repo.get_case(s, cid).indication == "keep me"
    assert current_revision(make_engine(empty)) == head_revision()


def test_the_baseline_can_be_undone(empty):
    init_db(empty)
    command.downgrade(config(empty), "base")
    engine = make_engine(empty)
    assert not set(BASELINE_COLUMNS) & set(inspect(engine).get_table_names())
    assert current_revision(engine) is None
    init_db(empty)                                                          # and applied again
    assert schema_differences(engine) == []


# ------------------------------------------------------------------ a database made before migrations existed
def test_a_database_from_before_migrations_is_taken_over_and_keeps_its_cases(empty):
    engine = make_engine(empty)
    Base.metadata.create_all(engine)                                        # exactly what init_db used to do
    with session_scope(empty) as s:
        cid = repo.create_case(s, indication="made before alembic").id
    assert current_revision(engine) is None
    init_db(empty)
    assert current_revision(engine) == head_revision()
    with session_scope(empty) as s:
        assert repo.get_case(s, cid).indication == "made before alembic"
    assert schema_differences(engine) == []


def test_a_database_that_does_not_match_the_baseline_is_refused_not_stamped(tmp_path):
    path = tmp_path / "odd.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cases (id TEXT PRIMARY KEY, indication TEXT)")        # missing most columns
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE llm_calls (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()
    url = f"sqlite:///{path.as_posix()}"
    with pytest.raises(SchemaMismatch, match="cases does not match the baseline"):
        init_db(url)
    assert "alembic_version" not in inspect(make_engine(url)).get_table_names()      # nothing was marked as known


def test_a_database_with_only_some_of_the_tables_is_refused(tmp_path):
    path = tmp_path / "half.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cases (id TEXT PRIMARY KEY)")
    con.commit()
    con.close()
    with pytest.raises(SchemaMismatch, match="events, llm_calls"):
        init_db(f"sqlite:///{path.as_posix()}")


def test_tables_dropped_behind_the_version_record_are_reported_not_ignored(tmp_path):
    url = f"sqlite:///{(tmp_path / 'gone.db').as_posix()}"
    init_db(url)
    Base.metadata.drop_all(make_engine(url))                               # the version record stays
    with pytest.raises(SchemaMismatch, match="missing"):
        init_db(url)


# ------------------------------------------------------------------ the files themselves
def test_the_baseline_renders_for_postgres_without_a_postgres_server():
    out = io.StringIO()
    cfg = config("postgresql+psycopg://user:pw@localhost:5432/none")
    cfg.output_buffer = out
    command.upgrade(cfg, "head", sql=True)
    sql = out.getvalue()
    assert "CREATE TABLE cases" in sql and "JSONB" in sql and "ON DELETE CASCADE" in sql
    assert "CREATE INDEX ix_events_case_id_id" in sql and "alembic_version" in sql


def test_the_baseline_does_not_import_the_models_because_it_must_stay_a_snapshot():
    tree = ast.parse((ROOT / "migrations" / "versions" / "0001_baseline.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree)
                                                                                      if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and m.split(".")[0] == "app" for m in imported), imported
    assert BASELINE == "0001"


def test_every_revision_has_a_working_downgrade():
    for path in (ROOT / "migrations" / "versions").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        body = re.search(r"def downgrade\(\) -> None:\n(.*)", text, re.S).group(1)
        assert body.strip() and body.strip() != "pass", f"{path.name} cannot be undone"


def test_the_image_carries_the_migrations_because_the_service_runs_them_at_startup():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    base = dockerfile.split("AS test")[0]
    assert "COPY alembic.ini" in base and "COPY migrations migrations" in base
    assert Path(ROOT / "alembic.ini").exists() and "alembic==" in (ROOT / "requirements-api.txt").read_text(encoding="utf-8")
