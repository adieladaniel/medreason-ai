# Running MedReason-AI in containers

Written 2026-10-27, after the stack was built and run for real on this laptop (Docker Desktop 4.49.0). The reasons for
each choice are next to it so the writeup can reuse them.

## What runs where

| Piece | Where | Why |
|---|---|---|
| API (FastAPI, the worker, all models) | container `api` | the code under test, one image |
| PostgreSQL 16 | container `postgres`, host port 5433 | cases, audit events, language-model calls. 5432 belongs to the PostgreSQL installed on Windows |
| Qdrant 1.19.1 | container `qdrant`, host port 6333 | the evidence index, as a server |
| Ollama | the Windows host, not a container | it already uses the GPU and holds the model; the API reaches it at `host.docker.internal:11434`, which Docker Desktop forwards to the host's localhost, so `OLLAMA_HOST` stays unset and Ollama stays off the network |

Every published port is bound to `127.0.0.1`. The API has no login, so it must never be reachable from the network.

## First-time setup (PowerShell, from the project folder)

```powershell
copy .env.example .env                       # then edit .env: set POSTGRES_PASSWORD (letters and digits only)
docker compose build api                     # a few minutes the first time (it downloads torch), 2.9 GB
docker compose up -d postgres qdrant
docker compose run --rm -e HF_HUB_OFFLINE=0 setup python -m app.prefetch    # downloads the pretrained models, 9.4 minutes when measured
docker compose run --rm setup python -m app.rag.index --build               # fills Qdrant with 811 passages
ollama serve                                 # in its own window, if it is not already running
docker compose up -d api
```

Then open http://localhost:8000/#/analyze, and http://localhost:8000/api/ready to see whether every dependency is up. Later
starts are `docker compose up -d` and later stops are `docker compose down` (the volumes, and so the models and the index,
are kept; `docker compose down -v` deletes them). Send a real validation case with `python scripts/submit_case.py --study-id 53383543`.

## Choices and their reasons

**One image, three stages (`Dockerfile`).** `base` holds the dependencies and the code. `runtime` adds a non-root user and the
start command. `test` adds the test suite and runs it during the build. `docker build --target test .` is the CI job
(`.github/workflows/tests.yml`), and because the build context leaves out everything git ignores (`.dockerignore`), it is also a
check that no test needs a file that exists only on this laptop. It runs the same tests as `python -m pytest`.

**CPU-only torch, installed first.** The default PyPI wheel carries CUDA libraries the container cannot use and makes the image
several gigabytes larger. The versions are the ones in `requirements.txt`; `requirements-api.txt` holds only what the service
imports, and `tests/test_packaging.py` fails if the two files disagree on a version or if the code imports a package the runtime
file does not install.

**The image holds code only.** The trained models and result files (`experiments/`) are mounted read-only; the pretrained
downloads live in two named volumes. Nothing derived from the MIMIC data is copied into an image layer, so the image can be
rebuilt, inspected or shared without that question coming up. The build context excludes `data/` and every model or prediction
file (`.dockerignore` mirrors `.gitignore`, and a test checks they agree).

**The container is offline.** `HF_HUB_OFFLINE=1`, so a service that handles clinical text never fetches anything at request time,
and a model that was not put in place fails loudly at startup. `python -m app.prefetch` is the one place the network is used. It
loads each model through the function the app uses, so what is downloaded is exactly what is read later.

**The `setup` service.** An opt-in profile of the same image with the open-access corpus mounted, for the two one-time steps. It is
not started by `docker compose up`, publishes no port and mounts no patient data. It exists because a relative `-v ./data/corpus`
on `docker compose run` silently mounted nothing here (the index build then failed on an empty corpus), whereas a path inside
the compose file is resolved correctly.

**Secrets.** The database password comes from `.env`, which git and docker both ignore. The compose file has no default for it
(`${POSTGRES_PASSWORD:?...}` refuses to start without one), and `.env.example` holds a placeholder.

**Non-root and a health check.** The process runs as user `app` (checked: `id` answers `app`, writing into `/app/experiments`
fails with "read-only file system"). The image's health check is the liveness endpoint only, so the container is not restarted
because Ollama is down. `/api/ready` is the check for whether a case can run.

## What was measured when it was run for real (2026-10-27)

- Image 2.87 GB. API container 1.77 GiB of its 5 GiB limit after warm-up and two cases; Postgres 49 MiB; Qdrant 57 MiB.
- `/api/ready` was `ok` with all four dependencies, the API reaching Ollama through `host.docker.internal` in 13 ms and the Qdrant
  server reporting its 811 passages.
- A pneumonia validation case finished in 52.5 s, of which 20 s was the first language-model call while Ollama loaded its model
  into the GPU (the next four took 0.7 to 1.2 s each). A normal case, which needs no evidence note, finished in 6.5 s.
- The JSON columns are `jsonb` in the container's Postgres, each case stored 10 audit events and its language-model calls, and the
  container's logs were JSON with the request id on the API lines and the case id on every line of the job.
- A restart of the API kept both cases and the uploaded image (served from the volume). Stopping Ollama gave `degraded`
  (HTTP 200); stopping Qdrant as well gave `degraded` in 3.8 s, inside the 5 s deadline; starting Qdrant again restored its data.

The same stack was then run on 16 validation cases twice (2026-10-28): every case matched its offline twin and the timing budget is
in `experiments/e2e/RESULTS.md`.

## Limits to know

- The build was checked here, but the GitHub Actions workflow could not be run from this laptop. Its steps use the standard
  checkout, buildx and build-push actions, and the same `docker build --target test .` runs locally.
- With `jsonb`, PostgreSQL does not keep the order of keys, so `timings` in a case comes back as `writing, verifying, explaining,
  predicting, retrieving` and not in stage order. The Analyze page looks stages up by name, so it is unaffected; a client that
  iterates the object should use its own stage order.
- The evidence index lives in a Qdrant volume built from the corpus on this laptop. A different machine needs the corpus
  (`python -m app.rag.build_corpus`) before the index step.
- Docker Desktop and the WSL machine share the 16 GB of RAM with Ollama and anything else open. WSL is capped at 8 GB; the stack
  needs about 2 GB of that plus Ollama's 1.2 GB of RAM and 2.4 GB of GPU memory on the host.
- Still not built: the GPU is not used for the models in the container (the local torch is CPU-only as well), no image is pushed
  anywhere, and no reverse proxy or TLS, which a deployment beyond this laptop would need before it handled any real data.
