"""scripts/smoke_stack.py against a real server on a real socket: the app, the worker thread and the database are the real ones,
only the stage functions are replaced. The script is what CI uses to judge a running stack, so what matters here is that it passes a
healthy one and fails, with a message that names the problem, for each way a stack can be wrong."""
import importlib.util
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from app.config import Settings
from app.main import create_app
from app.pipelines import staged
from app.pipelines.runner import JobRunner

spec = importlib.util.spec_from_file_location("smoke_stack", Path(__file__).resolve().parent.parent / "scripts" / "smoke_stack.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

PREDICTION = {"belief": {"predicted_class": "Pneumonia", "confidence": 0.9,
                         "probabilities": {"Normal": 0.05, "Pneumonia": 0.9, "Lung Cancer": 0.05}, "history": []},
              "consistency": {"status": "agree", "detail": "All agree."}, "missing_info_plan": {}, "modalities": {}, "seconds": {}}


def working_stages():
    return [staged.Stage("predicting", "prediction", lambda ctx: {**PREDICTION, "modalities_used": sorted(ctx["inputs"])})]


def broken_stages(message="no model"):
    """What a CI runner has: no trained models, so the first stage fails."""
    def predict(ctx):
        raise FileNotFoundError(2, message, "/app/experiments/blood_baseline/model.joblib")
    return [staged.Stage("predicting", "prediction", predict)]


@pytest.fixture
def serve(tmp_path):
    """Start the real app on a free port; returns its base URL. `stages=None` runs no worker at all."""
    servers = []

    def start(stages, **settings):
        settings.setdefault("ollama_url", "http://127.0.0.1:9")      # nothing listens there, so a test never depends on Ollama running
        url =f"sqlite:///{(tmp_path / f's{len(servers)}.db').as_posix()}"
        cfg = Settings(database_url=url, upload_dir=tmp_path / "uploads", run_worker=stages is not None, **settings)
        runner = JobRunner(url, stages=stages, deps={"no_figures": True}, figure_dir=cfg.upload_dir) if stages is not None else None
        app = create_app(cfg, runner, build_dashboard=False)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started, "the test server did not start"
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(10)


def run(base, **kw):
    lines = []
    ok = smoke.run(base, out=lines.append, wait=kw.pop("wait", 20), **kw)
    return ok, "\n".join(lines)


# ------------------------------------------------------------------ a healthy stack passes
def test_a_healthy_stack_passes_every_check_and_a_finished_case_is_deleted(serve):
    base = serve(working_stages())
    ok, report = run(base, expect="done")
    assert ok, report
    assert report.count("ok    ") == 6 and "FAIL" not in report
    assert "ended done" in report and "17 labs" in report and "degraded" in report        # Qdrant and Ollama are not running in a test
    assert smoke.call(base, "GET", "/api/cases?limit=100")[1].count(b'"id"') == 0          # the smoke case did not stay behind


def test_what_a_ci_runner_has_is_checked_for_not_excused_a_case_without_models_must_end_failed_and_say_nothing_about_the_server(serve):
    base = serve(broken_stages())
    ok, report = run(base, expect="failed")
    assert ok, report
    assert "ended failed" in report and "nothing leaked" in report


# ------------------------------------------------------------------ each way a stack can be wrong is reported
def test_a_case_that_ends_in_the_wrong_state_is_reported(serve):
    ok, report = run(serve(broken_stages()), expect="done")
    assert not ok and "FAIL  case: case ended 'failed', expected 'done'" in report
    ok, report = run(serve(working_stages()), expect="failed")
    assert not ok and "case ended 'done', expected 'failed'" in report


def test_strict_mode_fails_when_a_dependency_is_missing_and_names_it(serve):
    ok, report = run(serve(working_stages()), strict=True)
    assert not ok and "FAIL  ready: --strict, and not ok:" in report and "ollama" in report            # Ollama is never running in a test; Qdrant is, if a local index folder exists


def test_a_stack_started_without_a_worker_is_ready_by_design_but_fails_where_it_matters(serve):
    ok, report = run(serve(None))
    assert "ok    ready:" in report                                  # readiness says "no worker by configuration" on purpose
    assert not ok and "FAIL  case: POST /api/cases answered 503" in report and "without a worker" in report


def test_an_error_that_reveals_a_server_path_fails_the_check(serve, monkeypatch):
    """Undo the redaction on purpose and check the smoke test notices what the redaction exists to prevent."""
    from app.pipelines import runner
    monkeypatch.setattr(runner, "public_message", lambda exc: f"{type(exc).__name__}: {exc}")
    ok, report = run(serve(broken_stages()), expect="failed")
    assert not ok and "FAIL  case: the case view reveals" in report and "/app/" in report


def test_a_server_that_is_not_there_gives_one_failure_and_stops(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                                                            # closed again: nothing listens
    ok, report = run(f"http://127.0.0.1:{port}", wait=1)
    assert not ok and report.count("\n") == 0 and report.startswith("FAIL  health: /api/health did not answer within 1 s")


def test_a_refused_submission_that_is_stored_anyway_is_reported(serve, monkeypatch):
    base = serve(working_stages())
    real = smoke.call

    def lenient(base_, method, path, body=None, content_type=None, timeout=15):
        status, raw = real(base_, method, path, body, content_type, timeout)
        return (202, raw) if (method, path, status) == ("POST", "/api/cases", 422) else (status, raw)

    monkeypatch.setattr(smoke, "call", lenient)
    ok, report = run(base)
    assert not ok and "FAIL  bad input: an unknown lab gave 202" in report


def test_the_multipart_body_has_each_field_once_and_an_end_marker():
    body, ctype = smoke.form({"indication": "cough", "labs": "{}"})
    boundary = ctype.split("boundary=")[1]
    text = body.decode()
    assert text.count(f"--{boundary}\r\n") == 2 and text.endswith(f"--{boundary}--\r\n") and 'name="indication"\r\n\r\ncough' in text


# ------------------------------------------------------------------ the CI script and workflow (checked as text: CI itself needs Docker)
ROOT = Path(__file__).resolve().parent.parent
CI_SCRIPT = (ROOT / "scripts" / "ci_stack.sh").read_text(encoding="utf-8")


def test_the_ci_script_refuses_the_real_project_before_it_can_delete_any_volume():
    refuse = CI_SCRIPT.index('"$PROJECT" = "medreason"')
    assert refuse < CI_SCRIPT.index("down -v") and refuse < CI_SCRIPT.index("dc up") and 'exit 2' in CI_SCRIPT[refuse:refuse + 300]
    assert 'PROJECT="${PROJECT:-medreasonci}"' in CI_SCRIPT                          # and the default is not the real one


def test_the_ci_script_reaches_docker_only_through_its_own_project_and_a_throwaway_env_file():
    lines = [ln for ln in CI_SCRIPT.splitlines() if "docker compose" in ln and not ln.lstrip().startswith("#")]
    assert len(lines) == 1 and '-p "$PROJECT" --env-file "$ENV_FILE"' in lines[0], lines      # every other call goes through dc()
    assert "mktemp" in CI_SCRIPT and 'rm -f "$ENV_FILE"' in CI_SCRIPT and "trap cleanup EXIT" in CI_SCRIPT
    assert "POSTGRES_PASSWORD" in CI_SCRIPT and "/dev/urandom" in CI_SCRIPT                 # a random password, never a default one


def test_the_ci_script_checks_the_old_database_the_restart_and_runs_the_smoke_test_twice():
    assert CI_SCRIPT.index("create_all") < CI_SCRIPT.index("dc up -d --wait postgres qdrant api") < CI_SCRIPT.index("dc restart api")
    runs = [ln for ln in CI_SCRIPT.splitlines() if ln.startswith('"$PYTHON" scripts/smoke_stack.py')]
    assert len(runs) == 2 and all("--expect failed" in ln for ln in runs), runs
    for fact in ("select version_num from alembic_version", "made before migrations", "jsonb"):
        assert fact in CI_SCRIPT, fact


def test_compose_lets_ci_turn_warm_up_off_because_a_runner_has_no_models():
    import yaml
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["x-app"]["environment"]["MEDREASON_WARM_UP_MODELS"] == "${MEDREASON_WARM_UP_MODELS:-true}"     # on by default
    assert "MEDREASON_WARM_UP_MODELS=false" in CI_SCRIPT


def test_the_workflow_builds_the_runtime_image_and_runs_the_stack_script_without_building_twice():
    import yaml
    workflow = ROOT / ".github" / "workflows" / "tests.yml"
    if not workflow.exists():
        pytest.skip(".github is left out of the image build context (.dockerignore), and this suite also runs there")
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    assert set(jobs) == {"tests", "stack"}
    build, run = jobs["stack"]["steps"][-2:]
    assert build["with"]["target"] == "runtime" and build["with"]["load"] is True and build["with"]["tags"] == "medreason-api"
    assert run["run"] == "bash scripts/ci_stack.sh" and run["env"]["SKIP_BUILD"] == "1"


# ------------------------------------------------------------------ a server that lies about one thing at a time
def lying(monkeypatch, match, answer):
    """Make smoke.call return `answer` (status, bytes) for the calls where match(method, path) is true, and the real thing otherwise."""
    real = smoke.call

    def call(base, method, path, body=None, content_type=None, timeout=15):
        return answer if match(method, path) else real(base, method, path, body, content_type, timeout)

    monkeypatch.setattr(smoke, "call", call)


@pytest.mark.parametrize("match, answer, expected", [
    (lambda m, p: m == "DELETE", (204, b""), "FAIL  case: a deleted case can still be read"),
    (lambda m, p: m == "DELETE", (200, b""), "FAIL  case: DELETE did not answer 204"),
    (lambda m, p: p.endswith("/events"), (200, b'{"events": []}'), "FAIL  case: a finished case has no events"),
    (lambda m, p: p == "/api/ready", (503, b'{"status": "down", "checks": {"database": {"ok": false, "detail": "no connection"}, '
                                          b'"worker": {"ok": true, "detail": ""}}}'), "FAIL  ready: database is not ok: no connection"),
    (lambda m, p: p == "/api/ready", (200, b'{"status": "ok", "checks": {"database": {"ok": true}, "worker": {"ok": true}}}'), None),
    (lambda m, p: p == "/static/shell.js", (404, b""), "FAIL  pages: not served: /static/shell.js"),
    (lambda m, p: p == "/", (200, b"<html>nginx default page</html>"), "FAIL  pages: / answered 200 without the page"),
    (lambda m, p: p == "/api/input-spec", (200, b'{"labs": []}'), "FAIL  input-spec: status 200, 0 labs"),
    (lambda m, p: p == "/api/ready", (200, b"<html>not json</html>"), "answered 200 with something that is not JSON"),
], ids=["delete does nothing", "delete answers wrongly", "no events", "database down", "healthy ready", "asset missing", "wrong page",
        "wrong lab count", "ready is not json"])
def test_each_lie_is_reported_by_the_check_that_covers_it(serve, monkeypatch, match, answer, expected):
    base = serve(working_stages())
    lying(monkeypatch, match, answer)
    ok, report = run(base, expect="done")
    if expected is None:
        assert ok, report                                                  # the control: a true answer of the same shape passes
    else:
        assert not ok and expected in report, report


def test_a_case_that_never_finishes_is_reported_with_the_stage_it_is_stuck_in(serve):
    release = threading.Event()
    stages = [staged.Stage("predicting", "prediction", lambda ctx: release.wait(30) and PREDICTION)]
    base = serve(stages)
    try:
        ok, report = run(base, wait=3)
    finally:
        release.set()
    assert not ok and "FAIL  case: case " in report and "was still 'running' at stage 'predicting' after 3 s" in report


def test_a_submission_that_is_refused_but_stored_anyway_is_reported(serve, monkeypatch):
    base = serve(working_stages())
    real = smoke.call

    def call(base_, method, path, body=None, content_type=None, timeout=15):
        if method == "POST" and path == "/api/cases" and b"not_a_lab" in (body or b""):
            valid, ctype = smoke.form({"indication": "stored although refused"})
            real(base_, "POST", "/api/cases", valid, ctype, timeout)            # the case is saved ...
            return 422, b"{}"                                                    # ... and the client is told it was not
        return real(base_, method, path, body, content_type, timeout)

    monkeypatch.setattr(smoke, "call", call)
    ok, report = run(base)
    assert not ok and "FAIL  bad input: a refused submission was stored anyway" in report
