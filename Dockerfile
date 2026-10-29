# MedReason-AI service image. Three stages:
#   base     dependencies and the code. No data, no trained models, no pretrained weights.
#   test     base plus the test suite, run during the build. `docker build --target test .` is the CI job, and because
#            the build context leaves out everything git ignores (.dockerignore), it is also a check that the tests do
#            not depend on files that only exist on one laptop.
#   runtime  base plus a non-root user and the start command. This is the last stage, so a plain `docker build .` makes it.
#
# The trained models (experiments/) and the pretrained downloads are mounted by docker-compose.yml, not copied in, so the
# image holds nothing derived from the MIMIC data and can be rebuilt or shared without thinking about it.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Every package, with every dependency of a dependency, at one version and with the SHA-256 of its files: --require-hashes makes pip
# refuse anything else. The lock is made from requirements-api.txt by scripts/lock_dependencies.py. torch and torchvision are in it
# as the CPU-only wheels (2.11.0+cpu, from the second index), because the default wheel bundles about 3 GB of CUDA libraries the
# container cannot use. PyPI stays the first index; a name pinned as "+cpu" exists only on the PyTorch one.
COPY requirements-api.lock .
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-api.lock

COPY app app
COPY explain explain
COPY alembic.ini .
COPY migrations migrations


FROM base AS test
COPY requirements-test.lock .
RUN pip install --require-hashes -r requirements-test.lock
COPY pytest.ini requirements.txt requirements-api.txt requirements-test.txt Dockerfile docker-compose.yml .gitignore .dockerignore .env.example ./
COPY scripts scripts
COPY experiments experiments
COPY tests tests
RUN python -m pytest -p no:warnings 


FROM base AS runtime
# Non-root. These folders are created and owned here so the named volumes compose mounts on them start out writable.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/data/uploads /models/huggingface /home/app/.torchxrayvision \
 && chown -R app:app /app/data /models /home/app
USER app
ENV HF_HOME=/models/huggingface \
    MEDREASON_LOG_FORMAT=json
EXPOSE 8000
# Liveness only (/api/health never fails because Ollama or Qdrant is down). /api/ready is the check for "can it run a case".
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
