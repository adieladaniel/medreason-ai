"""
Liveness, readiness and usage numbers for whoever runs the service.

`/api/health` only says the process answers. `/api/ready` says whether it can do its job: it probes the
four things a case needs, all at once, so a dead dependency costs one wait and not the sum of them, and the
whole call is cut off at DEADLINE seconds whatever the sockets are doing.

  database  a trivial query on the configured database
  worker    the thread that runs cases is alive (and how many cases wait)
  qdrant    the evidence index. A server is asked for its collection and point count. The embedded index is
            only checked for its file, never opened, because embedded Qdrant allows one client per folder
            and a probe that opened it would lock the retriever out
  ollama    the server answers and the configured model has been pulled

The overall status is "ok", "degraded" or "down". Without the database or the worker no case can run, so
that is "down" and the HTTP status is 503. Without Qdrant or Ollama a case still gets its prediction and
explanations and only the evidence note is missing (the pipeline records that stage as failed and leaves the
case "partial"), so that is "degraded" and the HTTP status stays 200. A load balancer should keep sending
traffic to a degraded instance, and a person reading the page can see which part is missing.

Each check's detail names the failure but never the connection string, host or file path, because the
endpoint has no login. The full error goes to the log with the same wording a person would search for.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

from sqlalchemy import func, select, text

from app.config import Settings
from app.database.models import Case, Event
from app.database.session import session_scope
from app.rag.names import DEFAULT_MODEL, collection_name, embedded_index_file

log = logging.getLogger("medreason.health")

PROBE_TIMEOUT = 2.0          # seconds a socket may wait, per connection attempt
# Hard limit for the whole readiness call. The socket timeout is not enough: it covers one connection attempt and
# not name lookup. With Ollama stopped, the probe measured 4.0 s here against a 2 s timeout, and the log said
# "timed out" (not "refused"), which fits two attempts ("localhost" tried as IPv6, then IPv4) that each waited
# out the timeout. That explanation was not isolated.
DEADLINE = 5.0
CRITICAL = ("database", "worker")


class ProbeFailed(Exception):
    """A probe that reached its dependency and found it unfit. The message is safe to show."""


def _result(ok: bool, detail: str, seconds: float) -> dict:
    return {"ok": ok, "detail": detail, "seconds": round(seconds, 3)}


def _timed(name: str, fn: Callable[[], str]) -> dict:
    """Run one probe. It returns a detail string when healthy and raises when not."""
    t0 = time.perf_counter()
    try:
        detail = fn()
        return _result(True, detail, time.perf_counter() - t0)
    except Exception as exc:                                       # noqa: BLE001  (any failure means not ready)
        log.warning("readiness: %s check failed: %s: %s", name, type(exc).__name__, exc)
        return _result(False, _public_reason(exc), time.perf_counter() - t0)


def _public_reason(exc: Exception) -> str:
    if isinstance(exc, ProbeFailed):
        return str(exc)
    if isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError)):
        return f"not reachable ({type(exc).__name__})"
    return f"failed ({type(exc).__name__})"


def _get_json(url: str, timeout: float) -> dict:
    if not url.startswith(("http://", "https://")):
        raise ProbeFailed("the configured address is not an http or https URL")
    with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310  (scheme checked above)
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------------ probes
def check_database(url: str) -> str:
    with session_scope(url) as s:
        s.execute(text("SELECT 1"))
    return "answers a query"


def check_worker(runner) -> str:
    if runner is None:
        return "no worker by configuration, so cases are refused"
    if not runner.alive():
        raise ProbeFailed("the worker thread is not running")
    return f"running, {runner.pending()} case(s) waiting, models: {runner.warm_state}"


def check_qdrant(settings: Settings, model: str = DEFAULT_MODEL, timeout: float = PROBE_TIMEOUT) -> str:
    name = collection_name(model, settings)
    if settings.qdrant_url:
        try:
            info = _get_json(f"{settings.qdrant_url.rstrip('/')}/collections/{name}", timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:                                    # the server answered, the collection is not there
                raise ProbeFailed(f"server reachable but collection {name} does not exist "
                                  "(run: python -m app.rag.index --build)") from exc
            raise
        points = (info.get("result") or {}).get("points_count") or 0
        if points <= 0:
            raise ProbeFailed(f"server reachable but collection {name} is empty (run: python -m app.rag.index --build)")
        return f"server has {points} passages in {name}"
    f = embedded_index_file(model, settings)
    if not f.is_file() or f.stat().st_size == 0:
        raise ProbeFailed("embedded index not built (run: python -m app.rag.index --build)")
    return "embedded index file present (not opened, so its contents are not verified)"


def check_ollama(settings: Settings, timeout: float = PROBE_TIMEOUT) -> str:
    tags = _get_json(settings.ollama_url.rstrip("/") + "/api/tags", timeout)
    names = {m.get("name") for m in tags.get("models", [])}
    wanted = settings.ollama_model
    if wanted not in names and (":" in wanted or f"{wanted}:latest" not in names):
        raise ProbeFailed(f"reachable, but model {wanted} has not been pulled")
    return f"reachable, model {wanted} available"


# ------------------------------------------------------------------ readiness
def readiness(settings: Settings, runner) -> tuple[str, dict]:
    probes = {
        "database": lambda: check_database(settings.database_url),
        "worker": lambda: check_worker(runner),
        "qdrant": lambda: check_qdrant(settings),
        "ollama": lambda: check_ollama(settings),
    }
    pool = ThreadPoolExecutor(max_workers=len(probes))
    futures = {name: pool.submit(_timed, name, fn) for name, fn in probes.items()}
    start = time.perf_counter()
    checks = {}
    for name, future in futures.items():
        try:
            checks[name] = future.result(timeout=max(0.0, DEADLINE - (time.perf_counter() - start)))
        except FutureTimeout:
            log.warning("readiness: %s check did not answer within %.1f s", name, DEADLINE)
            checks[name] = _result(False, f"did not answer within {DEADLINE:g} s", DEADLINE)
    pool.shutdown(wait=False)             # a probe past the deadline finishes on its own; the answer does not wait for it
    if any(not checks[n]["ok"] for n in CRITICAL):
        status = "down"
    elif any(not c["ok"] for c in checks.values()):
        status = "degraded"
    else:
        status = "ok"
    return status, checks


# ------------------------------------------------------------------ usage numbers
def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. q is 0 to 1."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def usage_stats(url: str, pending: int | None = None) -> dict:
    """Counts by status and seconds per stage, from the cases and events still in the database (so only
    the retention window). Seconds are those of finished stages; a failed stage is counted separately."""
    with session_scope(url) as s:
        by_status = dict(s.execute(select(Case.status, func.count()).group_by(Case.status)).all())
        finished = s.execute(select(Event.stage, Event.seconds).where(
            Event.status == "finished", Event.seconds.is_not(None))).all()
        failed = dict(s.execute(select(Event.stage, func.count()).where(Event.status == "failed")
                                .group_by(Event.stage)).all())
    per_stage: dict[str, list[float]] = {}
    for stage, seconds in finished:
        per_stage.setdefault(stage, []).append(seconds)
    stages = {
        stage: {"n": len(v), "failed": failed.get(stage, 0), "p50": round(percentile(v, 0.5), 2),
                "p95": round(percentile(v, 0.95), 2), "max": round(max(v), 2)}
        for stage, v in per_stage.items()
    }
    for stage, n in failed.items():
        stages.setdefault(stage, {"n": 0, "failed": n})
    return {"cases": {"total": sum(by_status.values()), "by_status": by_status}, "queue_pending": pending,
            "stage_seconds": stages}
