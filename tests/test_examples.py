"""
The example-case API (app/api/examples.py): the menu on the Analyze page. Curated cases are validation studies kept in a git-ignored
folder; the tests build a small fake one. The built-in 'not an X-ray' examples are checked against the real, committed guard, because
their whole point is that the server refuses them.
"""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.pipelines.runner import JobRunner


def make_demo(folder, ids=("alpha", "beta-2")):
    for i in ids:
        (folder / i).mkdir(parents=True)
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (120, 120), dtype=np.uint8), "L").save(folder / i / "image.jpg", "JPEG")
    (folder / "manifest.json").write_text(json.dumps({"note": "x", "cases": [
        {"id": i, "title": f"Case {i}", "note": f"Look at {i}", "indication": f"indication of {i}", "labs": {"wbc": 9.1},
         "study_id": 987654321, "expected": {"true_label": "Lung Cancer", "predicted_class": "Pneumonia"}} for i in ids]}), encoding="utf-8")


@pytest.fixture
def client(tmp_path):
    made = []

    def make(demo=None, guard=False):
        url = f"sqlite:///{(tmp_path / 'e.db').as_posix()}"
        cfg = Settings(database_url=url, upload_dir=tmp_path / "up", demo_dir=demo or tmp_path / "no-demo", xray_guard=guard)
        c = TestClient(create_app(cfg, JobRunner(url, stages=[], deps={"no_figures": True}, figure_dir=cfg.upload_dir), build_dashboard=False))
        c.__enter__()
        made.append(c)
        return c
    yield make
    for c in made:
        c.__exit__(None, None, None)


def test_without_a_demo_folder_only_the_built_in_examples_are_offered(client):
    r = client().get("/api/examples").json()
    assert r["curated_available"] is False
    assert [e["id"] for e in r["examples"]] == ["not-an-xray-photo", "not-an-xray-noise"]
    assert {e["kind"] for e in r["examples"]} == {"built-in"}


def test_curated_cases_come_first_with_titles_and_notes(client, tmp_path):
    make_demo(tmp_path / "demo")
    r = client(tmp_path / "demo").get("/api/examples").json()
    assert r["curated_available"] is True
    assert [e["id"] for e in r["examples"]][:2] == ["alpha", "beta-2"]
    assert r["examples"][0] == {"id": "alpha", "title": "Case alpha", "note": "Look at alpha", "kind": "validation case"}


def test_an_example_returns_its_inputs_and_never_the_study_id_or_the_answer(client, tmp_path):
    make_demo(tmp_path / "demo")
    c = client(tmp_path / "demo")
    d = c.get("/api/examples/alpha").json()
    assert d["indication"] == "indication of alpha" and d["labs"] == {"wbc": 9.1} and d["has_image"] is True
    for path in ("/api/examples", "/api/examples/alpha"):
        text = c.get(path).text
        assert "987654321" not in text and "Lung Cancer" not in text and "true_label" not in text and "expected" not in text


def test_the_image_is_served_with_its_type(client, tmp_path):
    make_demo(tmp_path / "demo")
    r = client(tmp_path / "demo").get("/api/examples/alpha/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (120, 120)


@pytest.mark.parametrize("bad", ["nope", "..%2Fmanifest", "ALPHA", "a" * 80, ".hidden", "alpha%2F..%2Fbeta-2"])
def test_unknown_or_unsafe_ids_are_not_found(client, tmp_path, bad):
    make_demo(tmp_path / "demo")
    c = client(tmp_path / "demo")
    assert c.get(f"/api/examples/{bad}").status_code == 404
    assert c.get(f"/api/examples/{bad}/image").status_code == 404


def test_a_manifest_entry_without_an_image_or_with_a_bad_id_is_ignored(client, tmp_path):
    make_demo(tmp_path / "demo", ids=("alpha",))
    m = json.loads((tmp_path / "demo" / "manifest.json").read_text())
    m["cases"] += [{"id": "ghost", "title": "t", "note": "n"}, {"id": "../etc", "title": "t", "note": "n"}, {"title": "no id", "note": "n"}]
    (tmp_path / "demo" / "manifest.json").write_text(json.dumps(m))
    ids = [e["id"] for e in client(tmp_path / "demo").get("/api/examples").json()["examples"]]
    assert ids == ["alpha", "not-an-xray-photo", "not-an-xray-noise"]


def test_a_broken_manifest_does_not_break_the_menu(client, tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "manifest.json").write_text("{ not json")
    r = client(tmp_path / "demo").get("/api/examples")
    assert r.status_code == 200 and r.json()["curated_available"] is False


@pytest.mark.parametrize("example", ["not-an-xray-photo", "not-an-xray-noise"])
def test_the_built_in_pictures_are_refused_by_the_real_guard_and_accepted_when_it_is_off(client, example):
    img = client().get(f"/api/examples/{example}/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    refused = client(guard=True).post("/api/cases", files={"image": (f"{example}.png", img.content, "image/png")})
    assert refused.status_code == 422 and refused.json()["detail"][0]["field"] == "image"
    assert client(guard=False).post("/api/cases", files={"image": (f"{example}.png", img.content, "image/png")}).status_code == 202


def test_the_built_in_pictures_are_the_same_every_time(client):
    c = client()
    assert c.get("/api/examples/not-an-xray-noise/image").content == c.get("/api/examples/not-an-xray-noise/image").content


def test_the_two_built_in_pictures_exercise_different_cues_of_the_guard(client):
    c = client(guard=True)
    photo = Image.open(io.BytesIO(c.get("/api/examples/not-an-xray-photo/image").content))
    noise = Image.open(io.BytesIO(c.get("/api/examples/not-an-xray-noise/image").content))
    assert photo.mode == "RGB" and noise.mode == "L"
    why = {}
    for name, im in (("photo", photo), ("noise", noise)):
        buf = io.BytesIO()
        im.save(buf, "PNG")
        why[name] = c.post("/api/cases", files={"image": (f"{name}.png", buf.getvalue(), "image/png")}).json()["detail"][0]["message"]
    assert "colour photograph" in why["photo"] and "no large-scale structure" in why["noise"]
