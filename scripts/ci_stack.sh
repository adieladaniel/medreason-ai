#!/usr/bin/env bash
# Start the container stack the way a fresh machine would, and check it. CI runs this (.github/workflows/tests.yml, job "stack"), and
# so can you: it is the same script, which is the point (a CI step that cannot be run locally cannot be debugged locally).
#
# What it proves that the unit tests and the image build cannot:
#   - the image starts, with its migrations, static files and settings, against a real Postgres 16 and a Qdrant server
#   - a database made BEFORE migrations existed (the old create_all schema, with a row in it) is taken over by the migrations at
#     startup: version 0001, the row still there, JSONB columns. This is the upgrade path for anyone with an existing volume.
#   - restarting the API on the migrated database works (the "already at the newest version" path)
#   - scripts/smoke_stack.py passes against the running API, before and after the restart
#
# What it does not do: run the trained models or the language model. They derive from MIMIC and are kept out of git, so a CI runner
# has none, and a case must therefore end "failed" at its first stage (the smoke test expects that and checks the error is clean).
# Accuracy and timing with the real models are scripts/run_e2e.py and docs/DOCKER.md.
#
# Safety: it uses its own compose project (medreasonci), so its volumes are not the ones holding your cases, and it deletes only
# those when it finishes. It refuses to run as the real project. Needs Docker with compose v2, python3 and the built image
# (set SKIP_BUILD=1 when CI has already built it; otherwise it builds `api` first).
#
#   bash scripts/ci_stack.sh            (PYTHON=python on Windows; API_PORT=8010 if 8000 is taken)
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${PROJECT:-medreasonci}"
API_PORT="${API_PORT:-8000}"
PYTHON="${PYTHON:-python3}"
if [ "$PROJECT" = "medreason" ]; then
  echo "refusing to run as the project 'medreason': the cleanup deletes its volumes, which hold real cases" >&2
  exit 2
fi

ENV_FILE="$(mktemp)"
printf 'POSTGRES_PASSWORD=ci%s\nAPI_PORT=%s\nMEDREASON_WARM_UP_MODELS=false\n' \
  "$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')" "$API_PORT" > "$ENV_FILE"

dc() { docker compose -p "$PROJECT" --env-file "$ENV_FILE" "$@"; }
sql() { dc exec -T postgres psql -U medreason -d medreason -tA -c "$1" | tr -d '\r'; }
step() { printf '\n== %s\n' "$1"; }
expect() { # expect "what" "wanted" "got"
  if [ "$2" != "$3" ]; then echo "FAIL  $1: wanted '$2', got '$3'" >&2; exit 1; fi
  echo "ok    $1: $3"
}

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo; echo "== the stack failed; the last log lines of each service"
    dc logs --no-color --tail=120 || true
  fi
  dc down -v --remove-orphans > /dev/null 2>&1 || true
  rm -f "$ENV_FILE"
  exit "$status"
}
trap cleanup EXIT

LEGACY_ID=legacy00000000000000000000000001      # 32 characters, the width of a case id

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  step "build the runtime image"
  dc build api
fi

step "start Postgres"
dc up -d --wait postgres

step "make a database the way the service did before migrations existed (create_all, then one case)"
dc run --rm --no-deps -T api python - <<PY
from sqlalchemy import text
from app.config import get_settings
from app.database.models import Base
from app.database.session import make_engine
engine = make_engine(get_settings().database_url)
Base.metadata.create_all(engine)
with engine.begin() as conn:
    conn.execute(text("INSERT INTO cases (id, created_at, updated_at, status, stage, indication, timings) "
                      "VALUES ('$LEGACY_ID', now(), now(), 'done', 'done', 'made before migrations', '{}')"))
print("legacy schema created with one case")
PY
expect "no version record yet" "" "$(sql "select to_regclass('public.alembic_version')")"

step "start the stack (the API migrates the database at startup)"
dc up -d --wait postgres qdrant api
expect "database version" "0001" "$(sql 'select version_num from alembic_version')"
expect "the old case survived" "made before migrations" "$(sql "select indication from cases where id = '$LEGACY_ID'")"
expect "JSON columns are jsonb" "jsonb" "$(sql "select data_type from information_schema.columns where table_name = 'cases' and column_name = 'labs'")"

step "smoke test the running API"
"$PYTHON" scripts/smoke_stack.py --url "http://127.0.0.1:$API_PORT" --expect failed --wait 120

step "restart the API on the migrated database and test again"
dc restart api
"$PYTHON" scripts/smoke_stack.py --url "http://127.0.0.1:$API_PORT" --expect failed --wait 120
expect "database version after the restart" "0001" "$(sql 'select version_num from alembic_version')"
expect "the old case is still there" "1" "$(sql "select count(*) from cases where id = '$LEGACY_ID'")"
expect "cases left behind by the smoke test" "1" "$(sql 'select count(*) from cases')"

step "the whole stack checked out"
