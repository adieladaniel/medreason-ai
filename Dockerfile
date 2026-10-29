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

# CPU-only torch first: it is the largest layer and changes least. Versions must match requirements.txt.
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app app
COPY explain explain


FROM base AS test
RUN pip install pytest==9.1.1 httpx==0.28.1 PyYAML==6.0.3
COPY pytest.ini requirements.txt Dockerfile docker-compose.yml .gitignore .dockerignore .env.example ./
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
