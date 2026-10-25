"""The case API end to end, with a fake worker: real HTTP routes, real database, real upload handling, real
background thread, but stage functions that return canned results, so no model or service is needed."""
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.database import repository as repo
from app.database.session import init_db, session_scope
from app.main import create_app
from app.pipelines import staged
from app.pipelines.runner import JobRunner

PREDICTION = {"belief": {"predicted_class": "Pneumonia", "confidence": 0.9,
                         "probabilities": {"Normal": 0.05, "Pneumonia": 0.9, "Lung Cancer": 0.05}, "history": []},
              "consistency": {"status": "agree", "detail": "All three agree."}, "missing_info_plan": {},
              "modalities": {}, "seconds": {}}


def png(size=(128, 128), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("L", size, 90).save(buf, fmt)
    return buf.getvalue()


def fake_stages(gate: threading.Event | None = None, fail_explaining=False):
    """predicting, then explaining (writes a real figure file, optionally waits on `gate` or fails), then a
    last stage so a case has more than one result to show."""
    def predict(ctx):
        return {**PREDICTION, "modalities_used": sorted(ctx["inputs"])}

    def explain(ctx):
        ctx["progress"]({"blood": {"figure": None}})
        if gate is not None:
            assert gate.wait(10), "the test never released the worker"
        if fail_explaining:
            raise RuntimeError("explainer broke")
        ctx["figure_dir"].mkdir(parents=True, exist_ok=True)
        (ctx["figure_dir"] / "blood.png").write_bytes(png())
        return {"blood": {"figure": f"{ctx['case_id']}/blood.png", "matches_prediction": True}}

    def write(ctx):
        return {"plan": {"queries": []}, "passages": [], "unanswered": [], "final": {"text": ["done"]}}

    return [staged.Stage("predicting", "prediction", predict), staged.Stage("explaining", "explanations", explain),
            staged.Stage("writing", "evidence", write)]


@pytest.fixture
def make_client(tmp_path):
    started = []

    def make(stages=None, **settings):
        url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
        cfg = Settings(database_url=url, upload_dir=tmp_path / "uploads", **settings)
        runner = JobRunner(url, stages=stages or fake_stages(), deps={"no_figures": True}, figure_dir=cfg.upload_dir)
        app = create_app(cfg, runner, build_dashboard=False)
        app.state.poll_interval = 0.02
        client = TestClient(app)
        client.__enter__()
        started.append(client)
        client.upload_dir, client.db_url = cfg.upload_dir, url
        return client

    yield make
    for c in started:
        c.__exit__(None, None, None)


def wait_for(client, case_id, statuses=("done", "partial", "failed"), timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        view = client.get(f"/api/cases/{case_id}").json()
        if view["status"] in statuses:
            return view
        time.sleep(0.02)
    raise AssertionError(f"case {case_id} never reached {statuses}; last view: {view['status']} / {view['stage']}")


def submit(client, **kw):
    data = {k: v for k, v in kw.items() if k in ("indication", "labs") and v is not None}
    files = {"image": ("film.png", kw["image"], "image/png")} if kw.get("image") else None
    return client.post("/api/cases", data=data, files=files)


# ------------------------------------------------------------------ submitting and reading
def test_a_case_is_accepted_at_once_and_finishes_in_the_background(make_client):
    c = make_client()
    r = submit(c, indication="fever and cough", labs=json.dumps({"wbc": 15.2, "albumin": 2.8}))
    assert r.status_code == 202
    body = r.json()
    assert len(body["id"]) == 32 and body["status"] == "queued" and r.headers["location"] == f"/api/cases/{body['id']}"
    assert set(body["links"]) == {"self", "events", "stream"}
    view = wait_for(c, body["id"])
    assert view["status"] == "done" and view["stage"] == "done" and view["error"] is None
    assert view["prediction"]["belief"]["predicted_class"] == "Pneumonia"
    assert view["prediction"]["modalities_used"] == ["blood", "text"]
    assert view["inputs"]["labs"] == {"wbc": 15.2, "albumin": 2.8} and view["inputs"]["indication"] == "fever and cough"
    assert set(view["timings"]) == {"predicting", "explaining", "writing"}


def test_an_xray_is_stored_and_served_back_without_revealing_server_paths(make_client, tmp_path):
    c = make_client()
    data = png((200, 150))
    r = submit(c, image=data)
    assert r.status_code == 202
    cid = r.json()["id"]
    view = wait_for(c, cid)
    text = json.dumps(view)
    assert view["inputs"]["image"] is True and view["inputs"]["image_url"] == f"/api/cases/{cid}/image"
    assert str(tmp_path) not in text and "image_path" not in text and "input.png" not in text
    assert (c.upload_dir / cid / "input.png").read_bytes() == data
    got = c.get(f"/api/cases/{cid}/image")
    assert got.status_code == 200 and got.content == data
    with session_scope(c.db_url) as s:
        assert repo.get_case(s, cid).image_sha256 and len(repo.get_case(s, cid).image_sha256) == 64


def test_an_empty_file_field_from_a_browser_form_counts_as_no_image(make_client):
    c = make_client()
    r = c.post("/api/cases", data={"indication": "cough and fever"}, files={"image": ("", b"", "application/octet-stream")})
    assert r.status_code == 202
    assert wait_for(c, r.json()["id"])["inputs"]["image"] is False


def test_figures_are_served_by_url_and_only_by_known_name(make_client):
    c = make_client()
    cid = submit(c, indication="fever").json()["id"]
    view = wait_for(c, cid)
    assert view["explanations"]["blood"]["figure"] == f"/api/cases/{cid}/figures/blood.png"
    got = c.get(view["explanations"]["blood"]["figure"])
    assert got.status_code == 200 and got.headers["content-type"] == "image/png" and got.content.startswith(b"\x89PNG")
    assert c.get(f"/api/cases/{cid}/figures/text.png").status_code == 404           # a valid name, but never written
    assert c.get(f"/api/cases/{cid}/figures/secret.png").status_code in (404, 422)


def test_unknown_and_malformed_ids_are_not_found(make_client):
    c = make_client()
    for path in ("/api/cases/" + "0" * 32, "/api/cases/not-an-id", "/api/cases/..%2f..%2fetc", "/api/cases/" + "0" * 32 + "/events",
                 "/api/cases/" + "0" * 32 + "/image", "/api/cases/" + "0" * 32 + "/llm-calls"):
        assert c.get(path).status_code in (404, 422), path


def test_recent_cases_are_listed_newest_first_with_their_headline(make_client):
    c = make_client()
    ids = [submit(c, indication=f"case {i} fever").json()["id"] for i in range(3)]
    for i in ids:
        wait_for(c, i)
    rows = c.get("/api/cases?limit=2").json()["cases"]
    assert [r["id"] for r in rows] == ids[::-1][:2]
    assert rows[0]["predicted_class"] == "Pneumonia" and rows[0]["status"] == "done" and rows[0]["confidence"] == 0.9
    assert c.get("/api/cases?limit=0").status_code == 422


# ------------------------------------------------------------------ refusing bad input
@pytest.mark.parametrize("kw,field", [
    ({}, "form"),
    ({"indication": "   "}, "form"),
    ({"indication": "???"}, "form"),
    ({"labs": "{}"}, "form"),
    ({"labs": "not json"}, "labs"),
    ({"labs": json.dumps({"wbcc": 5})}, "labs"),
    ({"labs": json.dumps({"wbc": -3})}, "labs"),
    ({"indication": "x" * 2001}, "indication"),
])
def test_bad_or_empty_requests_get_422_naming_the_field(make_client, kw, field):
    c = make_client()
    r = submit(c, **kw)
    assert r.status_code == 422 and r.json()["detail"][0]["field"] == field and r.json()["detail"][0]["message"]
    assert c.get("/api/cases").json()["cases"] == []                       # nothing was stored


@pytest.mark.parametrize("data,filename,fragment", [
    (b"just some text", "film.png", "not an image"),
    (png(fmt="GIF"), "film.png", "GIF"),
    (png((20, 20)), "film.png", "at least 64"),
])
def test_bad_uploads_are_refused_and_leave_nothing_behind(make_client, data, filename, fragment):
    c = make_client()
    r = c.post("/api/cases", data={"indication": "fever"}, files={"image": (filename, data, "image/png")})
    assert r.status_code == 422 and fragment in r.json()["detail"][0]["message"]
    assert c.get("/api/cases").json()["cases"] == [] and not list(c.upload_dir.glob("*"))


def test_an_oversize_upload_is_refused_with_413(make_client):
    c = make_client(max_upload_mb=1)
    r = c.post("/api/cases", data={"indication": "fever"}, files={"image": ("big.png", b"\x00" * (1024 * 1024 + 5), "image/png")})
    assert r.status_code == 413 and "1 MB" in r.json()["detail"][0]["message"]


def test_a_decompression_bomb_is_refused_before_decoding(make_client):
    c = make_client(max_image_pixels=10_000)
    r = submit(c, image=png((300, 300)))
    assert r.status_code == 422 and "10,000 pixels" in r.json()["detail"][0]["message"]


def test_the_server_says_so_when_too_many_cases_are_waiting(make_client):
    gate = threading.Event()
    c = make_client(stages=fake_stages(gate), max_queue=2)
    first = submit(c, indication="fever one").json()["id"]
    wait_for(c, first, statuses=("running",))
    submit(c, indication="fever two")
    submit(c, indication="fever three")
    r = submit(c, indication="fever four")
    assert r.status_code == 429 and r.headers["retry-after"] == "60"
    gate.set()
    wait_for(c, first)


def test_without_a_worker_the_server_refuses_cases_honestly(tmp_path):
    cfg = Settings(database_url=f"sqlite:///{(tmp_path / 'x.db').as_posix()}", upload_dir=tmp_path / "up", run_worker=False)
    with TestClient(create_app(cfg, build_dashboard=False)) as c:
        assert c.post("/api/cases", data={"indication": "fever"}).status_code == 503


# ------------------------------------------------------------------ progress, failure, audit
def test_results_are_visible_while_the_case_is_still_running(make_client):
    gate = threading.Event()
    c = make_client(stages=fake_stages(gate))
    cid = submit(c, indication="fever").json()["id"]
    view = wait_for(c, cid, statuses=("running",))
    while not view["explanations"]:                      # the stage name changes a moment before the first piece is saved
        view = c.get(f"/api/cases/{cid}").json()
        time.sleep(0.02)
    assert view["stage"] == "explaining"
    assert view["status"] == "running" and view["prediction"]["belief"]["predicted_class"] == "Pneumonia"
    assert view["explanations"] == {"blood": {"figure": None}}                 # the piece saved by `progress`
    assert view["evidence"] is None
    gate.set()
    assert wait_for(c, cid)["status"] == "done"


def test_a_failing_stage_gives_a_partial_case_with_its_prediction(make_client):
    c = make_client(stages=fake_stages(fail_explaining=True))
    cid = submit(c, indication="fever").json()["id"]
    view = wait_for(c, cid)
    assert view["status"] == "partial" and "explaining failed" in view["error"] and "explainer broke" in view["error"]
    assert view["prediction"]["belief"]["predicted_class"] == "Pneumonia"


def test_the_audit_trail_lists_every_stage_with_seconds(make_client):
    c = make_client()
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid)
    events = c.get(f"/api/cases/{cid}/events").json()["events"]
    assert [(e["stage"], e["status"]) for e in events] == [(s, st) for s in ("predicting", "explaining", "writing") for st in ("started", "finished")]
    assert all(e["seconds"] is not None for e in events if e["status"] == "finished")
    later = c.get(f"/api/cases/{cid}/events?after={events[2]['id']}").json()["events"]
    assert [e["id"] for e in later] == [e["id"] for e in events[3:]]


def sse(client, url):
    """Reads a server-sent-event stream to its end and returns [(event name, parsed data)]."""
    out, name = [], None
    with client.stream("GET", url) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                out.append((name, json.loads(line.split(":", 1)[1])))
    return out


def test_the_stream_replays_what_happened_and_ends_when_the_case_finishes(make_client):
    c = make_client()
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid)
    got = sse(c, f"/api/cases/{cid}/stream")                 # connecting after the fact still gets everything
    names = [n for n, _ in got]
    assert names.count("stage") == 6 and names[-1] == "done" and "case" in names
    assert got[-1][1] == {"status": "done", "error": None}
    final_case = [d for n, d in got if n == "case"][-1]
    assert final_case["status"] == "done" and final_case["prediction"]["belief"]["predicted_class"] == "Pneumonia"


def test_the_stream_follows_a_running_case_to_the_end(make_client):
    gate = threading.Event()
    c = make_client(stages=fake_stages(gate))
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid, statuses=("running",))
    threading.Timer(0.3, gate.set).start()                    # released while the client is already listening
    got = sse(c, f"/api/cases/{cid}/stream")
    stages = [(d["stage"], d["status"]) for n, d in got if n == "stage"]
    assert stages[0] == ("predicting", "started") and stages[-1] == ("writing", "finished")
    assert [d["status"] for n, d in got if n == "case"][0] in ("running", "queued") and got[-1][0] == "done"


def test_every_model_call_can_be_read_back(make_client):
    c = make_client()
    cid = submit(c, indication="fever").json()["id"]
    wait_for(c, cid)
    with session_scope(c.db_url) as s:
        repo.add_llm_call(s, repo.get_case(s, cid), model="llama3.2:latest", purpose="background_sentence",
                          prompt="P", response="R", seconds=1.5)
    calls = c.get(f"/api/cases/{cid}/llm-calls").json()["calls"]
    assert [(x["model"], x["prompt"], x["response"], x["seconds"]) for x in calls] == [("llama3.2:latest", "P", "R", 1.5)]


# ------------------------------------------------------------------ deleting and expiring
def test_a_finished_case_can_be_deleted_with_its_files_and_a_running_one_cannot(make_client):
    gate = threading.Event()
    c = make_client(stages=fake_stages(gate))
    cid = submit(c, image=png(), indication="fever").json()["id"]
    wait_for(c, cid, statuses=("running",))
    assert c.delete(f"/api/cases/{cid}").status_code == 409
    gate.set()
    wait_for(c, cid)
    assert (c.upload_dir / cid / "input.png").exists() and (c.upload_dir / cid / "blood.png").exists()
    assert c.delete(f"/api/cases/{cid}").status_code == 204
    assert c.get(f"/api/cases/{cid}").status_code == 404 and not (c.upload_dir / cid).exists()
    assert c.delete(f"/api/cases/{cid}").status_code == 404


def test_cases_a_dead_process_left_unfinished_are_failed_at_startup(tmp_path):
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    init_db(url)
    with session_scope(url) as s:
        c = repo.create_case(s, indication="was running when the server died")
        repo.set_stage(s, c, "explaining")
        cid = c.id
    cfg = Settings(database_url=url, upload_dir=tmp_path / "uploads")
    with TestClient(create_app(cfg, JobRunner(url, stages=fake_stages()), build_dashboard=False)) as client:
        view = client.get(f"/api/cases/{cid}").json()
    assert view["status"] == "failed" and "restart" in view["error"]


def test_expired_cases_and_their_files_are_removed_at_startup(tmp_path):
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    init_db(url)
    uploads = tmp_path / "uploads"
    with session_scope(url) as s:
        old = repo.create_case(s, indication="old")
        old.created_at = datetime.now(timezone.utc) - timedelta(days=60)
        old.status = "done"
        fresh = repo.create_case(s, indication="fresh")
        fresh.status = "done"
        old_id, fresh_id = old.id, fresh.id
    for cid in (old_id, fresh_id):
        (uploads / cid).mkdir(parents=True)
        (uploads / cid / "blood.png").write_bytes(b"png")
    cfg = Settings(database_url=url, upload_dir=uploads, retention_days=30)
    with TestClient(create_app(cfg, JobRunner(url, stages=fake_stages()), build_dashboard=False)) as client:
        assert client.get(f"/api/cases/{old_id}").status_code == 404
        assert client.get(f"/api/cases/{fresh_id}").status_code == 200
    assert not (uploads / old_id).exists() and (uploads / fresh_id / "blood.png").exists()


def test_the_dashboard_routes_still_work(make_client):
    c = make_client()
    assert c.get("/api/health").json() == {"status": "ok"}
    assert c.get("/").status_code == 200 and "MedReason" in c.get("/").text


def test_the_image_endpoint_never_serves_a_file_from_outside_the_upload_folder(make_client, tmp_path):
    """Cases made by the command line point at files elsewhere; whatever the database says, only the upload
    folder is served."""
    c = make_client()
    outside = tmp_path / "secret.png"
    outside.write_bytes(png())
    with session_scope(c.db_url) as s:
        cid = repo.create_case(s, indication="fever", image_path=str(outside)).id
    assert c.get(f"/api/cases/{cid}/image").status_code == 404
    inside = c.upload_dir / cid
    inside.mkdir(parents=True)
    (inside / "input.png").write_bytes(png())
    with session_scope(c.db_url) as s:
        repo.get_case(s, cid).image_path = str(inside / "input.png")
    assert c.get(f"/api/cases/{cid}/image").status_code == 200


def test_the_input_spec_endpoint_describes_the_form(make_client):
    c = make_client(max_upload_mb=3)
    spec = c.get("/api/input-spec").json()
    assert spec["image"]["max_mb"] == 3 and len(spec["labs"]) == 17 and spec["lab_groups"][0] == "Blood count"
    assert c.post("/api/cases", data={"labs": json.dumps({spec["labs"][0]["name"]: 5})}).status_code == 202   # every listed lab is accepted


def test_every_time_the_api_sends_is_marked_as_utc(make_client):
    """A time without a zone is read as local time by a browser, which showed elapsed times hours wrong."""
    c = make_client()
    cid = submit(c, indication="fever").json()["id"]
    view = wait_for(c, cid)
    with session_scope(c.db_url) as s:
        repo.add_llm_call(s, repo.get_case(s, cid), model="m", purpose="p", prompt="P", response="R", seconds=1)
    stamps = [view["created_at"], view["updated_at"], *[e["ts"] for e in view["events"]],
              c.get("/api/cases").json()["cases"][0]["created_at"], c.get(f"/api/cases/{cid}/llm-calls").json()["calls"][0]["ts"]]
    assert all(t.endswith("+00:00") for t in stamps), stamps
    age = datetime.now(timezone.utc) - datetime.fromisoformat(view["created_at"])
    assert timedelta(0) <= age < timedelta(minutes=2)


# ------------------------------------------------------------------ the helper that sends a validation case
def test_the_case_helper_sends_what_the_form_would_and_leaves_dropped_inputs_out(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    import submit_case
    film = tmp_path / "x.jpg"
    film.write_bytes(png(fmt="JPEG"))
    case = {"indication": "fever", "labs": {"wbc": 12.5, "crp": None}, "image_path": str(film), "true_label": "Pneumonia"}
    data, files = submit_case.build_request(case, set())
    assert data == {"indication": "fever", "labs": json.dumps({"wbc": 12.5})} and files["image"][0] == "film.jpg"
    data, files = submit_case.build_request(case, {"image", "blood"})
    assert data == {"indication": "fever"} and files is None


def test_the_case_helper_does_not_offer_the_sealed_test_split():
    import subprocess, sys
    r = subprocess.run([sys.executable, "scripts/submit_case.py", "--study-id", "1", "--split", "test"], capture_output=True, text=True)
    assert r.returncode != 0 and "invalid choice" in r.stderr


def test_what_the_helper_sends_is_accepted_by_the_api(make_client, tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    import submit_case
    film = tmp_path / "x.jpg"
    film.write_bytes(png(fmt="JPEG"))
    data, files = submit_case.build_request({"indication": "fever", "labs": {"wbc": 12.5}, "image_path": str(film)}, set())
    c = make_client()
    r = c.post("/api/cases", data=data, files=files)
    assert r.status_code == 202
    assert wait_for(c, r.json()["id"])["prediction"]["modalities_used"] == ["blood", "image", "text"]
