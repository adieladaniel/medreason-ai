"""scripts/rehearse_demo.py against a real server on a real socket, with stage functions that return what the manifest froze (or something
else, to see the script notice). The script is what is run before a demo, so what matters is that it passes when the demo will work and
says which case and which field when it will not."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests.test_smoke_stack import serve  # noqa: F401  (fixture)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("rehearse_demo", SCRIPTS / "rehearse_demo.py")
rehearse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearse)
smoke = rehearse.smoke

from app.pipelines import staged  # noqa: E402

EXPECTED = {
    "Alpha": {"true_label": "Normal", "predicted_class": "Normal", "pooled_confidence": 0.906, "consistency": "agree",
              "inputs": ["blood", "image", "text"], "low_confidence": False, "prediction_set": ["Normal"], "cancer_alert": False},
    "Beta": {"true_label": "Lung Cancer", "predicted_class": "Pneumonia", "pooled_confidence": 0.461, "consistency": "conflict",
             "inputs": ["image", "text"], "low_confidence": True, "prediction_set": ["Pneumonia", "Lung Cancer"], "cancer_alert": True},
}


def stored_prediction(e):
    """What stage_predict stores, built from a frozen `expected` block."""
    rest = round((1 - e["pooled_confidence"]) / 2, 4)
    probs = {c: rest for c in ("Normal", "Pneumonia", "Lung Cancer")}
    probs[e["predicted_class"]] = e["pooled_confidence"]
    return {"belief": {"predicted_class": e["predicted_class"], "confidence": e["pooled_confidence"], "probabilities": probs, "history": []},
            "consistency": {"status": e["consistency"], "detail": ""}, "missing_info_plan": {}, "modalities": {}, "seconds": {},
            "decision": {"low_confidence": e["low_confidence"], "prediction_set": e["prediction_set"],
                         "cancer_alert": {"informative": True, "flag": e["cancer_alert"]}},
            "modalities_used": e["inputs"]}


def fake_stages(change=None):
    """Answer each case with its frozen values, found by the indication text; `change` edits one case's answer."""
    def predict(ctx):
        key = ctx["inputs"]["text"]
        e = dict(EXPECTED[key])
        if change and key in change:
            e.update(change[key])
        stored = stored_prediction(e)
        stored["modalities_used"] = sorted(ctx["inputs"]) if not (change and "inputs" in change.get(key, {})) else e["inputs"]   # what really arrived
        return stored
    return [staged.Stage("predicting", "prediction", predict)]


@pytest.fixture
def demo(tmp_path):
    """A demo folder with the two cases and a manifest, as scripts/build_demo_cases.py writes it."""
    folder = tmp_path / "demo"
    cases = []
    for cid, key, labs in (("clear-normal", "Alpha", {"wbc": 9.0}), ("missed-cancer-alert-fires", "Beta", {})):
        (folder / cid).mkdir(parents=True)
        Image.new("L", (96, 96), 120).save(folder / cid / "image.jpg", "JPEG")
        cases.append({"id": cid, "title": cid, "question": "q", "note": "n", "indication": key, "labs": labs, "expected": EXPECTED[key]})
    (folder / "manifest.json").write_text(json.dumps({"note": "test", "cases": cases}), encoding="utf-8")
    return folder


def go(base, demo, **kw):
    """Run the whole rehearsal. The pictures check is stubbed here (it needs the guard on, which would also refuse the tiny test images
    used as films) and is tested on its own below."""
    lines, real = [], rehearse.check_builtin_pictures_are_refused
    rehearse.check_builtin_pictures_are_refused = lambda base_, *a: "stubbed in this test"
    try:
        ok = rehearse.run(base, demo / "manifest.json", allow_degraded=kw.pop("allow_degraded", True), wait=kw.pop("wait", 20), out=lines.append, **kw)
    finally:
        rehearse.check_builtin_pictures_are_refused = real
    return ok, "\n".join(lines)


# ------------------------------------------------------------------ the pure comparison
def view_of(e, **over):
    view = {"status": "done", "stage": "done", "error": None, "prediction": stored_prediction(e)}
    view.update(over)
    return view


def test_a_case_that_shows_what_was_frozen_has_no_problems():
    assert rehearse.compare(EXPECTED["Alpha"], view_of(EXPECTED["Alpha"])) == []
    assert rehearse.compare(EXPECTED["Beta"], view_of(EXPECTED["Beta"])) == []


@pytest.mark.parametrize("change, fragment", [
    ({"predicted_class": "Pneumonia"}, "predicted_class is 'Pneumonia', frozen as 'Normal'"),
    ({"consistency": "conflict"}, "consistency is 'conflict', frozen as 'agree'"),
    ({"inputs": ["image", "text"]}, "inputs is ['image', 'text']"),
    ({"low_confidence": True}, "low_confidence is True, frozen as False"),
    ({"cancer_alert": True}, "cancer_alert is True, frozen as False"),
    ({"prediction_set": ["Normal", "Pneumonia"]}, "prediction_set is ['Normal', 'Pneumonia']"),
    ({"pooled_confidence": 0.85}, "pooled_confidence is 0.850, frozen as 0.906"),
], ids=["class", "consistency", "inputs", "low confidence", "alert", "set", "confidence"])
def test_each_field_that_differs_is_named(change, fragment):
    problems = rehearse.compare(EXPECTED["Alpha"], view_of({**EXPECTED["Alpha"], **change}))
    assert any(fragment in p for p in problems), problems


def test_a_confidence_within_the_tolerance_is_not_a_difference_and_a_case_that_did_not_finish_is_one():
    nudged = {**EXPECTED["Alpha"], "pooled_confidence": 0.906 + rehearse.CONFIDENCE_TOLERANCE * 0.9}
    assert rehearse.compare(EXPECTED["Alpha"], view_of(nudged)) == []
    partial = rehearse.compare(EXPECTED["Alpha"], view_of(EXPECTED["Alpha"], status="partial", stage="writing", error="writing failed: Ollama"))
    assert partial and "ended 'partial' at stage 'writing': writing failed: Ollama" in partial[0]
    failed = rehearse.compare(EXPECTED["Alpha"], {"status": "failed", "stage": "predicting", "error": "boom", "prediction": None})
    assert "ended 'failed'" in failed[0] and any("predicted_class is None" in p for p in failed)


# ------------------------------------------------------------------ against a real server
def test_a_stack_that_shows_every_frozen_value_passes(serve, demo):
    base = serve(fake_stages(), demo_dir=demo)
    ok, report = go(base, demo)
    assert ok, report
    assert report.count("ok    clear-normal") == 1 and report.count("ok    missed-cancer-alert-fires") == 1
    assert "menu: all 2 cases offered" in report and report.endswith("the demo will show what its notes say")
    assert smoke.call(base, "GET", "/api/cases?limit=100")[1].count(b'"id"') == 0              # the rehearsal cases were deleted again


def test_keep_leaves_the_cases_for_looking_at_afterwards(serve, demo):
    base = serve(fake_stages(), demo_dir=demo)
    ok, _ = go(base, demo, keep=True)
    assert ok and len(smoke.case_ids(base)) == 2


def test_a_case_whose_answer_changed_is_named_with_the_field(serve, demo):
    base = serve(fake_stages({"Beta": {"cancer_alert": False}}), demo_dir=demo)
    ok, report = go(base, demo)
    assert not ok and "FAIL  missed-cancer-alert-fires" in report and "cancer_alert is False, frozen as True" in report
    assert "ok    clear-normal" in report and report.endswith("DO NOT DEMO until the lines marked FAIL are explained")


def test_strict_readiness_is_the_default_because_a_demo_without_evidence_is_not_the_demo(serve, demo):
    base = serve(fake_stages(), demo_dir=demo)
    ok, report = go(base, demo, allow_degraded=False)
    assert not ok and "--strict, and not ok: ollama" in report and "clear-normal" not in report      # it stops before sending anything


def test_a_server_that_does_not_offer_the_cases_says_the_folder_is_probably_not_mounted(serve, demo, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    base = serve(fake_stages(), demo_dir=empty)
    ok, report = go(base, demo)
    assert not ok and "does not offer these demo cases (is demo/cases mounted?): clear-normal, missed-cancer-alert-fires" in report


def test_a_missing_manifest_is_explained_not_a_traceback(serve, tmp_path):
    base = serve(fake_stages())
    ok, report = go(base, tmp_path / "no-such-folder")
    assert not ok and "cannot read the demo manifest" in report and "build_demo_cases.py" in report


def test_a_case_that_hangs_is_reported_with_its_stage(serve, demo):
    import threading
    release = threading.Event()
    stages = [staged.Stage("predicting", "prediction", lambda ctx: release.wait(30) and stored_prediction(EXPECTED["Alpha"]))]
    base = serve(stages, demo_dir=demo)
    try:
        ok, report = go(base, demo, wait=3)
    finally:
        release.set()
    assert not ok and "FAIL  clear-normal: still 'running' at stage 'predicting' after 3 s" in report


# ------------------------------------------------------------------ the pictures that are not X-rays
def test_the_built_in_pictures_are_refused_before_any_model_runs_when_the_guard_is_on(serve):
    base = serve(fake_stages(), xray_guard=True)
    assert "2 pictures refused before any model ran, nothing stored" in rehearse.check_builtin_pictures_are_refused(base)
    assert smoke.case_ids(base) == []


def test_a_server_with_the_guard_off_accepts_them_and_the_rehearsal_says_so(serve):
    base = serve(fake_stages(), xray_guard=False)
    with pytest.raises(smoke.Failed, match="was answered 202, a picture that is not an X-ray must be refused with 422"):
        rehearse.check_builtin_pictures_are_refused(base)


def test_a_menu_without_the_pictures_is_reported(serve):
    base = serve(fake_stages(), xray_guard=True)
    with pytest.raises(smoke.Failed, match="not offered by the example menu: not-an-xray-missing"):
        rehearse.check_builtin_pictures_are_refused(base, ("not-an-xray-photo", "not-an-xray-missing"))


def test_the_file_upload_part_of_the_multipart_body_carries_its_name_type_and_bytes():
    body, ctype = smoke.form({"indication": "cough"}, {"image": ("example.jpg", b"\xff\xd8binary\xff", "image/jpeg")})
    boundary = ctype.split("boundary=")[1].encode()
    assert body.count(b"--" + boundary + b"\r\n") == 2 and body.endswith(b"--" + boundary + b"--\r\n")
    assert b'name="image"; filename="example.jpg"\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8binary\xff\r\n' in body


def test_a_refused_picture_that_is_stored_anyway_is_reported(serve, monkeypatch):
    base = serve(fake_stages(), xray_guard=True)
    real = smoke.call

    def call(base_, method, path, body=None, content_type=None, timeout=15):
        if method == "POST" and path == "/api/cases":
            valid, ctype = smoke.form({"indication": "stored although refused"})
            real(base_, "POST", "/api/cases", valid, ctype, timeout)          # the case is saved ...
            return 422, b"{}"                                                   # ... and the client is told it was refused
        return real(base_, method, path, body, content_type, timeout)

    monkeypatch.setattr(smoke, "call", call)
    with pytest.raises(smoke.Failed, match="a refused picture was stored as a case"):
        rehearse.check_builtin_pictures_are_refused(base)
