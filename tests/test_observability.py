"""
Phase 7 step 5: logs that say which case a line belongs to, a readiness check that probes the dependencies,
and the usage numbers.

The dependency probes are tested against real HTTP servers started on a free local port (a few lines of
http.server that answer like Ollama or Qdrant would), not against patched functions, so the code that opens the
socket, reads the JSON and handles the failures is the code that runs. "Down" is a server that was started and
then closed, which is what a real stopped Ollama looks like: connection refused.
"""
import io
import json
import logging
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app import health
from app.config import Settings
from app.database import repository as repo
from app.database.session import init_db, session_scope
from app.main import create_app
from app.observability import bind, configure_logging, reset_logging
from app.pipelines import staged
from app.pipelines.runner import JobRunner, run_job
from app.rag.names import DEFAULT_MODEL, collection_name, embedded_index_file
from tests.test_api import fake_stages, make_client, submit, wait_for  # noqa: F401  (make_client is a fixture)


# ------------------------------------------------------------------ helpers
@pytest.fixture
def make_logs():
    """Install the real handler writing to memory. Returns a function giving the parsed JSON lines so far."""
    buf = io.StringIO()

    def install(level="INFO", fmt="json"):
        configure_logging(level, fmt, stream=buf)
        return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]

    install.buf = buf
    yield install
    reset_logging()


@pytest.fixture
def logs(make_logs):
    return make_logs()


class Stub:
    """A tiny HTTP server answering fixed JSON per path. `delay` holds each answer back."""

    def __init__(self, routes: dict, delay: float = 0.0):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                time.sleep(delay)
                code, body = routes.get(self.path, (404, {"status": {"error": "not found"}}))
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stubs():
    made = []

    def make(routes, delay=0.0):
        s = Stub(routes, delay)
        made.append(s)
        return s

    yield make
    for s in made:
        try:
            s.close()
        except Exception:                                          # already closed by the test
            pass


def ollama_routes(*models):
    return {"/api/tags": (200, {"models": [{"name": m} for m in models]})}


@pytest.fixture
def cfg(tmp_path):
    """Settings for readiness tests: a real SQLite file, an embedded index file in place, Ollama and Qdrant
    addresses filled in by each test."""
    c = Settings(database_url=f"sqlite:///{(tmp_path / 'ready.db').as_posix()}", upload_dir=tmp_path / "uploads",
                 qdrant_path=tmp_path / "qdrant", ollama_url="http://127.0.0.1:9", ollama_model="llama3.2:latest")
    init_db(c.database_url)
    f = embedded_index_file(DEFAULT_MODEL, c)
    f.parent.mkdir(parents=True)
    f.write_bytes(b"not empty")
    return c


@pytest.fixture
def runner(cfg):
    r = JobRunner(cfg.database_url, stages=[])
    r.start()
    yield r
    r.stop(timeout=5)


def new_case(url, **kw):
    init_db(url)
    with session_scope(url) as s:
        return repo.create_case(s, **kw).id


# ------------------------------------------------------------------ log context
def test_a_bound_case_id_is_on_every_line_in_the_block_and_gone_after(logs):
    log = logging.getLogger("medreason.test")
    log.info("before")
    with bind(case_id="abc"):
        log.info("inside")
        with bind(request_id="r1"):
            log.info("nested")
        log.info("back to one id")
    log.info("after")
    got = {x["message"]: (x["case_id"], x["request_id"]) for x in logs()}
    assert got == {"before": ("-", "-"), "inside": ("abc", "-"), "nested": ("abc", "r1"),
                   "back to one id": ("abc", "-"), "after": ("-", "-")}


def test_another_thread_does_not_see_the_case_id_bound_here(logs):
    seen = []

    def other():
        logging.getLogger("medreason.test").info("from the other thread")
        seen.append(True)

    with bind(case_id="mine"):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    (line,) = [x for x in logs() if x["message"] == "from the other thread"]
    assert line["case_id"] == "-"            # the worker's case id must never leak into a request, or the reverse


def test_json_lines_carry_extra_fields_and_exceptions(logs):
    log = logging.getLogger("medreason.test")
    log.info("stage done", extra={"stage": "predicting", "seconds": 1.25})
    try:
        raise ValueError("bad thing")
    except ValueError:
        log.exception("it broke")
    done, broke = logs()
    assert done["stage"] == "predicting" and done["seconds"] == 1.25 and done["level"] == "INFO"
    assert done["logger"] == "medreason.test" and done["ts"].endswith("+00:00")
    assert broke["level"] == "ERROR" and "ValueError: bad thing" in broke["exception"]


def test_text_format_shows_the_ids(make_logs):
    make_logs(fmt="text")
    with bind(case_id="abc123", request_id="req9"):
        logging.getLogger("medreason.test").warning("hello %s", "world")
    line = make_logs.buf.getvalue().strip()
    assert "WARNING" in line and "case=abc123" in line and "req=req9" in line and line.endswith("hello world")


def test_configuring_twice_does_not_print_each_line_twice(make_logs):
    make_logs()
    make_logs()
    logging.getLogger("medreason.test").info("once")
    assert len([x for x in make_logs.buf.getvalue().splitlines() if x]) == 1


def test_info_lines_are_visible_only_once_logging_is_configured(make_logs, capsys):
    logging.getLogger("medreason.test").info("invisible")      # no handler, default WARNING threshold
    assert capsys.readouterr().err == "" and make_logs.buf.getvalue() == ""
    get = make_logs()
    logging.getLogger("medreason.test").info("visible")
    assert [x["message"] for x in get()] == ["visible"]


def test_log_level_setting_filters_lines(make_logs):
    get = make_logs(level="WARNING")
    log = logging.getLogger("medreason.test")
    log.info("quiet")
    log.warning("loud")
    assert [x["message"] for x in get()] == ["loud"]


def test_settings_reject_unknown_log_options():
    with pytest.raises(ValueError):
        Settings(log_format="xml")
    with pytest.raises(ValueError):
        Settings(log_level="CHATTY")
    assert Settings(log_level="debug").log_level == "DEBUG"


# ------------------------------------------------------------------ the pipeline's log lines
def test_every_line_of_a_job_carries_its_case_id_even_from_code_that_knows_nothing_about_it(logs, tmp_path):
    url = f"sqlite:///{(tmp_path / 'j.db').as_posix()}"
    cid = new_case(url, indication="fever and cough")

    def predict(ctx):
        logging.getLogger("medreason.some.deep.helper").info("helper doing work")
        return {"x": 1}

    stages = [staged.Stage("predicting", "prediction", predict)]
    assert run_job(cid, url, stages, {"no_figures": True}, tmp_path) == "done"
    lines = logs()
    assert lines and all(x["case_id"] == cid for x in lines)
    messages = [x["message"] for x in lines]
    assert "helper doing work" in messages
    started = next(x for x in lines if x["message"].startswith("stage predicting started"))
    finished = next(x for x in lines if x["message"].startswith("stage predicting finished"))
    assert started["stage"] == finished["stage"] == "predicting" and finished["seconds"] >= 0
    last = lines[-1]
    assert last["message"].startswith("job finished: done") and last["status"] == "done"


def test_two_jobs_in_a_row_do_not_share_a_case_id(logs, tmp_path):
    url = f"sqlite:///{(tmp_path / 'j.db').as_posix()}"
    a, b = new_case(url, indication="one"), new_case(url, indication="two")
    stages = [staged.Stage("predicting", "prediction", lambda ctx: {"x": 1})]
    run_job(a, url, stages, {"no_figures": True}, tmp_path)
    run_job(b, url, stages, {"no_figures": True}, tmp_path)
    ids = [x["case_id"] for x in logs()]
    assert ids.count(a) == ids.count(b) > 0 and set(ids) == {a, b}
    assert ids.index(b) > max(i for i, v in enumerate(ids) if v == a)


def test_a_failing_stage_is_logged_as_an_error_with_the_case_and_stage(logs, tmp_path):
    url = f"sqlite:///{(tmp_path / 'j.db').as_posix()}"
    cid = new_case(url, indication="fever")

    def boom(ctx):
        raise RuntimeError("explainer broke")

    stages = [staged.Stage("predicting", "prediction", lambda ctx: {"x": 1}),
              staged.Stage("explaining", "explanations", boom)]
    assert run_job(cid, url, stages, {"no_figures": True}, tmp_path) == "partial"
    (err,) = [x for x in logs() if x["level"] == "ERROR"]
    assert err["case_id"] == cid and err["stage"] == "explaining" and "RuntimeError: explainer broke" in err["exception"]
    assert logs()[-1]["status"] == "partial"


def test_a_job_with_no_usable_input_is_logged_as_a_warning(logs, tmp_path):
    url = f"sqlite:///{(tmp_path / 'j.db').as_posix()}"
    cid = new_case(url, indication="???")                         # stored, but with no letters or digits it is not an input
    assert run_job(cid, url, [], {"no_figures": True}, tmp_path) == "failed"
    assert any(x["level"] == "WARNING" and x["case_id"] == cid for x in logs())


# ------------------------------------------------------------------ the request log
def test_every_response_carries_a_request_id_and_a_good_caller_id_is_kept(make_client):
    c = make_client()
    assert len(c.get("/api/health").headers["x-request-id"]) == 12
    assert c.get("/api/health", headers={"X-Request-ID": "trace-42.a_b"}).headers["x-request-id"] == "trace-42.a_b"
    bad = c.get("/api/health", headers={"X-Request-ID": "x" * 200}).headers["x-request-id"]
    assert bad != "x" * 200 and len(bad) == 12                      # too long: replaced, not echoed back
    spaced = c.get("/api/health", headers={"X-Request-ID": "has space"}).headers["x-request-id"]
    assert spaced != "has space"


def test_submitting_a_case_logs_it_under_its_own_id_and_never_logs_what_was_typed(make_client, make_logs):
    get = make_logs()
    c = make_client()
    secret = "zebrafish clinical secret words"
    r = submit(c, indication=secret, labs=json.dumps({"wbc": 13.37}))
    cid = r.json()["id"]
    wait_for(c, cid)
    lines = get()
    submitted = next(x for x in lines if x["message"].startswith("case submitted"))
    request_line = next(x for x in lines if x["message"].startswith("POST /api/cases ->"))
    assert submitted["case_id"] == cid and request_line["case_id"] == cid
    assert request_line["request_id"] == r.headers["x-request-id"] == submitted["request_id"]
    assert request_line["status_code"] == 202
    assert "labs: 1" in submitted["message"] and "indication: yes" in submitted["message"]
    assert secret not in make_logs.buf.getvalue() and "13.37" not in make_logs.buf.getvalue()
    # the worker's lines for this case are in the same log, tagged with the same id and no request id
    worker = [x for x in lines if x["logger"] == "medreason.runner"]
    assert worker and all(x["case_id"] == cid and x["request_id"] == "-" for x in worker)


def test_the_submit_line_says_how_many_cases_are_ahead_not_counting_itself(make_client, make_logs):
    get = make_logs()
    gate = threading.Event()
    c = make_client(stages=fake_stages(gate=gate))
    try:
        first = submit(c, indication="one").json()["id"]
        wait_for(c, first, statuses=("running",))                  # the worker has taken it and is held at the gate
        submit(c, indication="two")
        submit(c, indication="three")
    finally:
        gate.set()
    lines = [x["message"] for x in get() if x["message"].startswith("case submitted")]
    assert [m.rsplit(", ", 1)[1] for m in lines] == ["0 ahead in the queue)", "0 ahead in the queue)", "1 ahead in the queue)"]


def test_reads_are_quiet_by_default_changes_and_errors_are_not(make_client, make_logs):
    get = make_logs()
    c = make_client()
    c.get("/api/health")
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid)
    c.get(f"/api/cases/{'0' * 32}")                                  # 404
    r = submit(c)                                                    # 422, nothing to analyse
    assert r.status_code == 422
    paths = [x["message"] for x in get() if x["logger"] == "medreason.api"]
    assert not any(m.startswith("GET /api/health") for m in paths)
    assert not any(m.startswith(f"GET /api/cases/{cid}") for m in paths)       # the page polls, no line per poll
    assert any(m.startswith("POST /api/cases -> 202") for m in paths)
    assert any(m.startswith(f"GET /api/cases/{'0' * 32} -> 404") for m in paths)
    refused = [x for x in get() if x["message"].startswith("input refused")]
    assert refused and refused[0]["message"] == "input refused: field=form status=422"


def test_reads_are_logged_with_the_case_id_when_debug_is_on(make_client, make_logs):
    get = make_logs("DEBUG")
    c = make_client()
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid)
    c.get(f"/api/cases/{cid}/events")
    line = next(x for x in get() if x["message"].startswith(f"GET /api/cases/{cid}/events"))
    assert line["case_id"] == cid and line["level"] == "DEBUG"


def test_an_unhandled_error_is_logged_with_its_traceback(make_client, make_logs):
    get = make_logs()
    c = make_client()
    c.app.get("/api/boom")(lambda: 1 / 0)
    safe = TestClient(c.app, raise_server_exceptions=False)
    assert safe.get("/api/boom").status_code == 500
    lines = get()
    assert any(x["level"] == "ERROR" and "ZeroDivisionError" in x.get("exception", "") for x in lines)
    assert any(x["level"] == "WARNING" and x["message"].startswith("GET /api/boom -> 500") for x in lines)


# ------------------------------------------------------------------ readiness
def test_everything_up_is_ok(cfg, runner, stubs):
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest", "qwen2.5:3b")).url
    status, checks = health.readiness(cfg, runner)
    assert status == "ok" and all(c["ok"] for c in checks.values())
    assert set(checks) == {"database", "worker", "qdrant", "ollama"}
    assert "llama3.2:latest" in checks["ollama"]["detail"] and "0 case(s) waiting" in checks["worker"]["detail"]


def test_a_stopped_ollama_is_degraded_not_down(cfg, runner, stubs):
    stub = stubs(ollama_routes("llama3.2:latest"))
    cfg.ollama_url = stub.url
    stub.close()                                                     # now connection refused, like a stopped server
    status, checks = health.readiness(cfg, runner)
    assert status == "degraded"
    assert not checks["ollama"]["ok"] and "not reachable" in checks["ollama"]["detail"]
    assert all(checks[n]["ok"] for n in ("database", "worker", "qdrant"))
    assert cfg.ollama_url not in json.dumps(checks)                  # the address is not echoed to an anonymous caller


def test_ollama_without_the_model_is_degraded_and_says_which_model(cfg, runner, stubs):
    cfg.ollama_url = stubs(ollama_routes("mistral:latest")).url
    status, checks = health.readiness(cfg, runner)
    assert status == "degraded" and "llama3.2:latest has not been pulled" in checks["ollama"]["detail"]


def test_a_model_name_without_a_tag_matches_latest(cfg, stubs):
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest")).url
    cfg.ollama_model = "llama3.2"
    assert "available" in health.check_ollama(cfg)
    cfg.ollama_model = "llama3.2:1b"                                 # a tagged name must match exactly
    with pytest.raises(health.ProbeFailed):
        health.check_ollama(cfg)


def test_a_slow_dependency_times_out_instead_of_hanging_the_check(cfg, stubs):
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest"), delay=1.5).url
    t0 = time.perf_counter()
    with pytest.raises(Exception) as err:
        health.check_ollama(cfg, timeout=0.2)
    assert time.perf_counter() - t0 < 1.0
    assert health._public_reason(err.value).startswith("not reachable")


def test_the_probes_run_together_so_two_slow_ones_cost_one_wait(cfg, runner, stubs):
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest"), delay=0.7).url
    cfg.qdrant_url = stubs({f"/collections/{collection_name(DEFAULT_MODEL, Settings(qdrant_url='http://x'))}":
                            (200, {"result": {"points_count": 811}})}, delay=0.7).url
    t0 = time.perf_counter()
    status, checks = health.readiness(cfg, runner)
    elapsed = time.perf_counter() - t0
    assert status == "ok" and elapsed < 1.3, f"took {elapsed:.2f} s, one after another would be 1.4 s or more"


def test_a_probe_that_stalls_past_the_deadline_cannot_hold_the_answer(cfg, runner, stubs, monkeypatch):
    monkeypatch.setattr(health, "DEADLINE", 0.5)
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest"), delay=1.5).url      # answers, but long after the deadline
    t0 = time.perf_counter()
    status, checks = health.readiness(cfg, runner)
    assert time.perf_counter() - t0 < 1.2
    assert status == "degraded" and not checks["ollama"]["ok"] and checks["ollama"]["detail"] == "did not answer within 0.5 s"
    assert all(checks[n]["ok"] for n in ("database", "worker", "qdrant"))        # the quick ones are not lost


def test_qdrant_server_mode_reports_points_and_distinguishes_empty_missing_and_down(cfg, stubs):
    name = collection_name(DEFAULT_MODEL, Settings(qdrant_url="http://x"))
    server = stubs({f"/collections/{name}": (200, {"result": {"points_count": 811}})})
    cfg.qdrant_url = server.url
    assert health.check_qdrant(cfg) == f"server has 811 passages in {name}"

    empty = stubs({f"/collections/{name}": (200, {"result": {"points_count": 0}})})
    cfg.qdrant_url = empty.url
    with pytest.raises(health.ProbeFailed, match="is empty"):
        health.check_qdrant(cfg)

    missing = stubs({})                                              # answers 404 for every path
    cfg.qdrant_url = missing.url
    with pytest.raises(health.ProbeFailed, match="does not exist"):
        health.check_qdrant(cfg)

    dead = stubs({})
    cfg.qdrant_url = dead.url
    dead.close()
    status, checks = health.readiness(cfg, JobRunner(cfg.database_url, stages=[]))
    assert "not reachable" in checks["qdrant"]["detail"] and cfg.qdrant_url not in json.dumps(checks)


def test_the_embedded_index_is_checked_by_its_file_and_never_opened(cfg, tmp_path):
    before = sorted(p for p in (tmp_path / "qdrant").rglob("*"))
    detail = health.check_qdrant(cfg)
    assert "embedded index file present" in detail and str(tmp_path) not in detail and "storage.sqlite" not in detail
    assert sorted(p for p in (tmp_path / "qdrant").rglob("*")) == before     # embedded Qdrant would add a lock file if opened
    embedded_index_file(DEFAULT_MODEL, cfg).unlink()
    status, checks = health.readiness(cfg, JobRunner(cfg.database_url, stages=[]))
    assert not checks["qdrant"]["ok"] and "not built" in checks["qdrant"]["detail"]
    assert str(tmp_path) not in json.dumps(checks)                   # no server file paths in a response


def test_a_missing_database_is_down(cfg, runner, tmp_path, stubs):
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest")).url
    cfg.database_url = f"sqlite:///{(tmp_path / 'no' / 'such' / 'dir' / 'x.db').as_posix()}"   # parent missing, cannot open
    status, checks = health.readiness(cfg, runner)
    assert status == "down" and not checks["database"]["ok"]
    assert "x.db" not in json.dumps(checks)


def test_a_dead_worker_is_down(cfg, runner, stubs):
    cfg.ollama_url = stubs(ollama_routes("llama3.2:latest")).url
    runner.stop(timeout=5)
    status, checks = health.readiness(cfg, runner)
    assert status == "down" and checks["worker"]["detail"] == "the worker thread is not running"


def test_no_worker_by_configuration_is_reported_not_failed(cfg):
    assert "no worker by configuration" in health.check_worker(None)


def test_a_probe_failure_is_logged_with_the_reason(cfg, runner, logs):
    health.readiness(cfg, runner)                                    # Ollama points at a closed port
    warn = next(x for x in logs() if x["message"].startswith("readiness: ollama check failed"))
    assert warn["level"] == "WARNING" and warn["logger"] == "medreason.health"


def test_ready_endpoint_status_codes(make_client, monkeypatch, stubs):
    c = make_client()
    c.app.state.settings.ollama_url = stubs(ollama_routes("llama3.2:latest")).url
    monkeypatch.setattr(health, "check_qdrant", lambda *_a, **_k: "fine")
    r = c.get("/api/ready")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    monkeypatch.setattr(health, "check_qdrant", lambda *_a, **_k: (_ for _ in ()).throw(health.ProbeFailed("index gone")))
    r = c.get("/api/ready")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["checks"]["qdrant"]["detail"] == "index gone"

    monkeypatch.setattr(health, "check_database", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("db gone")))
    r = c.get("/api/ready")
    assert r.status_code == 503 and r.json()["status"] == "down"
    assert r.json()["checks"]["database"]["detail"] == "failed (RuntimeError)"


def test_liveness_does_not_depend_on_anything(make_client, monkeypatch):
    c = make_client()
    monkeypatch.setattr(health, "check_database", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("db gone")))
    assert c.get("/api/health").json() == {"status": "ok"}


def test_importing_the_health_module_does_not_load_the_models():
    code = "import sys, app.health; bad =[m for m in ('sentence_transformers', 'transformers') if m in sys.modules]; sys.exit(1 if bad else 0)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]


# ------------------------------------------------------------------ usage numbers
def test_percentile_interpolates_between_ranks():
    assert health.percentile([4.0], 0.95) == 4.0
    assert health.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert health.percentile([5, 1, 3, 2, 4], 0.95) == pytest.approx(4.8)
    assert health.percentile([10, 20], 0.5) == 15
    with pytest.raises(ValueError):
        health.percentile([], 0.5)


def test_usage_stats_counts_cases_and_summarises_finished_stage_seconds(tmp_path):
    url = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    init_db(url)
    with session_scope(url) as s:
        for status in ("done", "done", "partial", "failed"):
            case = repo.create_case(s, indication="x")
            case.status = status
            for seconds in (1, 2, 3):
                repo.add_event(s, case, "predicting", "finished", seconds)
            repo.add_event(s, case, "writing", "started")
        repo.add_event(s, case, "writing", "failed", 9.0)
    stats = health.usage_stats(url, pending=2)
    assert stats["cases"] == {"total": 4, "by_status": {"done": 2, "partial": 1, "failed": 1}}
    assert stats["queue_pending"] == 2
    p = stats["stage_seconds"]["predicting"]
    assert p["n"] == 12 and p["p50"] == 2.0 and p["max"] == 3 and p["failed"] == 0
    assert stats["stage_seconds"]["writing"] == {"n": 0, "failed": 1}      # a stage that only ever failed still shows up


def test_stats_endpoint_on_an_empty_database_and_after_a_case(make_client):
    c = make_client()
    empty = c.get("/api/stats").json()
    assert empty["cases"]["total"] == 0 and empty["stage_seconds"] == {} and empty["queue_pending"] == 0
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid)
    got = c.get("/api/stats").json()
    assert got["cases"]["by_status"] == {"done": 1}
    assert set(got["stage_seconds"]) == {"predicting", "explaining", "writing"}
    assert all(v["n"] == 1 for v in got["stage_seconds"].values())
