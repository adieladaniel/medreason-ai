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
quick; the slow part is the language model (measured as about 3 seconds per passage, of which about 2
seconds turned out to be a connection delay fixed in step 5) and the entailment checker (3
to 20 seconds a case), roughly 35 seconds warm and 89 seconds cold when measured. The staged design still holds, with
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
   Qdrant and Ollama. (done 2026-10-26)
6. Containers: Dockerfile, compose with the API, Postgres and Qdrant, Qdrant server mode, a CI
   workflow that runs the tests. Ollama stays on the Windows host (it already uses the GPU and holds
   the model), and the API reaches it through a setting instead of the hardcoded localhost address. (done 2026-10-27)
7. End-to-end run on real cases and a timing budget. (done 2026-10-28)

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

## Seeing what the service is doing (step 5, built 2026-10-26)

**The case id travels with the code, not through its arguments.** The worker runs one case while the API answers
requests on other threads, so a plain log mixes their lines. `app/observability.bind` puts the case id (and a request
id) in a context variable, and a filter on the log handler copies it onto every record. The pipeline's helpers, which know
nothing about cases, are tagged anyway, and a thread started elsewhere starts with no id, so the worker's case never
appears on a request's lines (both tested). Without this, every function that logs would need a case id parameter.

**Two formats, one switch.** Text for a terminal, one JSON object per line for a log collector (time, level, logger,
message, case id, request id, and fields such as stage and seconds). Only the `medreason` logger is configured. Before
this the runner's INFO lines were not visible at all under uvicorn, because nothing had installed a handler for them.

**Logs say what happened, the database says what was said.** A line records that a stage finished in so many seconds and
which inputs a case had. It never contains the typed indication, lab values or model output (tested by submitting a
case with distinctive text and searching the whole log). The prompts and answers of every language-model call are already
stored per case in the `llm_calls` table, behind the same retention rule as the case.

**Liveness and readiness are different questions.** `/api/health` answers whether the process is up, and never fails
because a dependency is down, so a supervisor does not restart a healthy process that is merely waiting for Ollama.
`/api/ready` answers whether a case can run: it probes the database (a query), the worker thread, the evidence index and
Ollama (reachable, and the configured model pulled). The result has three levels. `down` (HTTP 503) means the database or
the worker is gone, so no case can run. `degraded` (HTTP 200) means Qdrant or Ollama is missing: the prediction and the
explanations still work and only the written evidence note is lost, which the pipeline already handles by ending the case
as "partial". A load balancer should keep sending traffic to a degraded instance.

**What a probe is allowed to say.** The endpoint has no login, so a failure says what failed ("not reachable", "model not
pulled") and never the address, connection string or file path; the full reason goes to the log. Embedded Qdrant allows one
client per folder, so the probe checks that the index file exists and does not open it, and it says so in its answer; a
Qdrant server is asked for its collection and point count. The probes run at the same time and the whole call is cut off at 5
seconds, because a socket timeout covers one connection attempt and not name lookup.

**Usage numbers come from the database, not a metrics server.** `/api/stats` gives cases by status and the median, 95th
percentile and worst seconds of each stage, from the events table. A Prometheus server and a dashboard for it would be two
more services for one machine. The cost is stated: the numbers cover only the retention window and are not a time series.

**Two things measured by running it for real.** With Ollama stopped, the readiness call took 4.0 s against a 2 s timeout,
which is why it has the overall deadline. Chasing that showed that on this Windows laptop a connection to a closed loopback
port takes 2.0 s to be refused, and Python tries the IPv6 address (`::1`) before the IPv4 one, so every call to
`http://localhost:11434` paid about 2 s before reaching Ollama (measured 2.05 s, against 0.01 to 0.03 s for `127.0.0.1`).
The writer makes one such call per passage. The default address is now `127.0.0.1`, and on the same real case the mean time of
a language-model call went from 2.97 s to 0.84 s (five calls each, after warm-up) and the writing stage from 15.1 s to 4.2 s.
The earlier "about 3 seconds per passage" in these notes included that delay.

## Containers (step 6, built 2026-10-27)

The choices, the commands and what was measured are in `docs/DOCKER.md`. The ones that shape the design: the image holds code
and nothing derived from the clinical data (trained models are mounted read-only, pretrained downloads live in volumes, the
container runs offline); the test suite runs inside the image build, so CI and a clean-checkout check are one command; the
requirements the service needs are a pinned subset of the full list, with a test that fails if they drift or if the code imports
something the subset does not install; every port is published on 127.0.0.1 because there is no login; and the database
password has no default.

## The end-to-end run (step 7, 2026-10-28)

The method and numbers are in `experiments/e2e/RESULTS.md`. The design point worth keeping: each case is checked against an offline twin,
the stored per-modality probabilities pooled with the app's own rule for the inputs that were really sent. That turns "it ran" into
"the deployed pipeline reproduces the evaluated models", and it would have caught a preprocessing difference between the container and
the training runs. It did not: 32 of 32 runs matched, with a largest probability difference of 0.0001. The cases are chosen by rule
and a fixed seed so nobody can say they were picked to look good, and each is deleted from the server after it is read, because it holds
a real X-ray. The measured budget: the prediction is on screen in about half a second, a verified note takes about 15 to 20 seconds,
and verifying (about 3.4 seconds per drafted sentence) is the largest stage.

## Not in scope

Authentication and multi-user accounts (this is a demonstration, not a clinical service), a
message broker, horizontal scaling, and any claim of clinical use. Uploaded images are for
demonstration cases only.
