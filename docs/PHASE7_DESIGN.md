# Phase 7 design: backend and API

Written 2026-10-14, before building. Each choice has its reason next to it so the writeup can
reuse them. The status of each step is in ROADMAP.md.

## What the backend has to do

Accept a case (an X-ray, the clinical indication, optional labs), run the pipeline, and let a
browser watch it finish. The pipeline is slow and heavy: three models, a text explainer that
takes 60 to 90 seconds, a cross-encoder reranker, a local language model on a 4 GB GPU and an
entailment checker that takes seconds per sentence. It runs on one laptop.

## Decisions

**Prediction first, explanations after.** Only the text explainer is slow, and the code already
has a cheap text prediction (`predict_proba_texts`). So the first response, in about 10 seconds,
is the fused prediction, the agreement check and the missing-information plan. The explanations,
evidence and written note follow as they finish. A user waits 10 seconds for the answer instead of
two minutes. This needs a staged pipeline, which `app/agent/orchestrator.run_case` is not, because
it computes prediction and explanation together per modality.

**A job, not a long request.** `POST /api/cases` stores the case and returns 202 with an id. A
worker runs it in the background. The client reads progress and results from the database. A
request that stays open for two minutes breaks on proxies and timeouts, and cannot survive a
page reload.

**One in-process worker, not Celery and Redis.** There is one machine, one GPU and about 15 GB of
RAM, and the models do not fit twice. A queue service would add two moving parts to run a queue
of length one. The job state lives in the database, so the design is the same shape as a real
queue, and swapping the runner later does not change the API. The cost is stated: jobs do not
survive a restart, so at startup any unfinished case is marked failed (tested).

**SQLAlchemy, SQLite by default, Postgres in compose.** The same models and queries run on both
and the difference is one setting, `MEDREASON_DATABASE_URL`. SQLite makes the tests and local
runs need no service. Foreign keys and write-ahead logging are switched on for SQLite, because it
enforces neither foreign keys nor concurrent reads by default. JSON columns are JSONB on
Postgres. Docker is not installed on this machine, so the Postgres and compose parts cannot be
run here and will be marked untested until they are.

**Results are saved stage by stage.** The prediction is written the moment it exists, in its own
column, so a client can read it while the explanations are still running.

**An events table and an LLM-call table.** Events record each stage start, finish, failure and
duration, which drives the progress view and is the audit trail of a run. The LLM table stores
every prompt, response and duration. That is the tracing of language-model calls, kept in the
database already in use instead of a separate tracing server, which would be one more service for
a project that must run for free.

**Retention.** A case holds clinical text, lab values and an image. `purge_older_than` deletes
cases, their events, their LLM calls and the uploaded file after `retention_days` (default 30).

**Settings in one place.** `app/config.py` reads environment variables, so the laptop and the
container differ by configuration, not code.

## Order of work

1. Storage: models, sessions, repository, tests. (done)
2. Staged pipeline and the job runner, with restart recovery.
3. API: upload validation (size, real image, type), endpoints, progress stream.
4. Analyze page in the existing dashboard.
5. Observability: structured logs with case ids, a readiness check that probes the database,
   Qdrant and Ollama.
6. Containers: Dockerfile, compose with Postgres, Qdrant and Ollama, Qdrant server mode, a CI
   workflow that runs the tests.
7. End-to-end run on real cases and a timing budget.

## Not in scope

Authentication and multi-user accounts (this is a demonstration, not a clinical service), a
message broker, horizontal scaling, and any claim of clinical use. Uploaded images are for
demonstration cases only.
