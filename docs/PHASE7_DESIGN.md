# Phase 7 design: backend and API

Written 2026-10-14, before building. Each choice has its reason next to it so the writeup can
reuse them. The status of each step is in ROADMAP.md.

## What the backend has to do

Accept a case (an X-ray, the clinical indication, optional labs), run the pipeline, and let a
browser watch it finish. The pipeline is slow and heavy: three models, a text explainer, a cross-encoder reranker, a local language model on a 4 GB GPU and an
entailment checker that takes seconds per sentence. It runs on one laptop.

## Decisions

**Prediction first, explanations after.** The first response is the fused prediction, the agreement
check and the missing-information plan, from the models' probabilities alone. The explanations, evidence
and written note follow as they finish. This was designed on the belief that the text SHAP step took 60
to 90 seconds. Measured later, it takes under a second for a typical indication, so the explanations are
quick; the slow part is the language model (about 3 seconds per passage) and the entailment checker (3
to 20 seconds a case), roughly 35 seconds warm and 89 seconds cold. The staged design still holds, with
the prediction ready in under a second warm, but the reason is the evidence stages, not SHAP. This needs a staged pipeline, which `app/agent/orchestrator.run_case` is not, because
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
Postgres. The storage tests also ran against a real Postgres 16 container (10 of 10 passed, JSON
columns are `jsonb`). The compose file itself is not written or run yet.

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
4. Analyze page in the existing dashboard. (done)
5. Observability: structured logs with case ids, a readiness check that probes the database,
   Qdrant and Ollama.
6. Containers: Dockerfile, compose with the API, Postgres and Qdrant, Qdrant server mode, a CI
   workflow that runs the tests. Ollama stays on the Windows host (it already uses the GPU and holds
   the model), and the API reaches it through a setting instead of the hardcoded localhost address.
7. End-to-end run on real cases and a timing budget.

## The case API (step 3, built 2026-10-25)

**Endpoints.** `POST /api/cases` (multipart form with an optional image, indication and labs) returns 202 and the
id. `GET /api/cases/{id}` is the case as stored right now, so a client can poll. `GET /api/cases/{id}/stream` is
the same thing as server-sent events, for a page that wants to update as stages finish. The audit trail
(`/events`), every language-model prompt and answer (`/llm-calls`), the explanation figures, the uploaded image,
a list of recent cases and a delete complete it. FastAPI generates the interactive documentation at `/docs`.

**Polling a database, not a message bus.** The stream reads the database twice a second and sends what changed.
That is the same state the plain GET shows, so the two cannot disagree, a client that connects late gets the
whole history first, and a server restart loses nothing but the job in flight. The cost is a read every half
second per open stream, which is nothing for one machine.

**Checks before storing.** The upload is the only input a client fully controls, so it is the strictest: the
format is decided from the file's bytes, the size is capped while reading, the pixel count is checked before
decoding (a small PNG can unpack to gigabytes), and the whole image is decoded once so a truncated file fails
at the door and not halfway through a job. Lab names must be the 17 the model knows, because a typo that is
silently ignored would give a prediction without the value the user thought they sent. An over-long indication
is refused, not cut, because cutting changes what the model reads without saying so.

**Limits stated honestly.** No login, because this is a local demonstration with no real patients; cases expire
after 30 days and can be deleted. At most 10 cases wait for the worker, and past that the answer is 429, because
one worker means a longer queue only means a longer wait. Nothing checks that an uploaded image is a chest
X-ray. And the shown text for the same case can differ between runs, because the language model is not
reproducible even at temperature 0 (the checker still decides what is shown).

## The Analyze page (step 4, built 2026-10-25)

**Built from the API's own description of its inputs.** The form asks `/api/input-spec` for the lab names, units
and limits, so the page cannot offer a lab the server refuses or state a limit the server does not enforce. The
units matter more than they look: the blood model was trained on MIMIC's units, and a value typed in another unit
gives a wrong prediction with no error, so the units are read from the lab data and shown next to every field.

**Each section appears when its stage has stored something.** The page draws from the same case view the plain GET
returns, so what the stream shows and what a reload shows cannot differ. The prediction is on screen while the
explanations still say "computing", which is the point of staging the pipeline.

**Honest presentation.** A prediction below 0.55 confidence is flagged as a weak guess where it is headlined, because
that is where the reliability analysis declines to answer. The literature note says in the page itself that it is
background from papers, not proof about this patient, and every cited sentence has its source passage one click away,
because the checker can miss a sentence that mixes true facts into a false claim. The page gives no clinical advice,
and the one sentence of the agent layer that did was reworded.

**Plain files, no build step.** Vanilla JavaScript on the dashboard's existing helpers, like the rest of it, with all
server and paper text escaped before display. The front end is tested in a real browser with fake models, which found
bugs no unit test could (an element id shadowing a form method, a display rule overriding the hidden attribute, a
layout wider than a phone).

## Saying how sure the system is (2026-10-26)

**The models diagnose; this only reports confidence.** `app/agent/decision.py` reads the pooled probabilities and a small rules
file and returns whether confidence is low, which classes cannot be ruled out, and whether the Lung Cancer probability is high
enough for an alert. It has no predicted class of its own and a test fuzzes that it never changes its input.

**Rules come from a file, built from training data only.** `scripts/build_decision_rules.py` fits everything on the out-of-fold
training predictions and uses validation once, to report what each rule does. The app reads the committed file, because the
training predictions are git-ignored and absent from a deployment. A slow test checks the file is exactly what the script
produces.

**Calibrated per combination of inputs.** Pooling two models gives probabilities that behave differently from pooling three,
and a case without labs is common, so each of the seven combinations has its own rules. Using the all-inputs rule for a case
with one input missing would be unjustified.

**A separate alert, not a replacement for the answer.** Highest-probability-wins finds 50 percent of validation cancers where
the text model alone finds 61 percent. A cancer alert at a chosen cost in false alarms is shown beside the headline instead of
changing it. How many false alarms to accept is a product decision, so it is a setting, defaulting to 60 percent recall (13
alerts per 100 cases, about one in four real). An alert that was barely better than chance on validation (image alone) is not
shown, and the page says why.

**Honest limits.** The alert levels and the 2.0 lift cut-off are choices. The hard validation case that prompted this still does
not trigger the default alert (fused cancer probability 0.22 against 0.30), and is headlined Uncertain instead. The rules are
fitted on one dataset with 774 training cancers.

## Not in scope

Authentication and multi-user accounts (this is a demonstration, not a clinical service), a
message broker, horizontal scaling, and any claim of clinical use. Uploaded images are for
demonstration cases only.
