"""
scripts/build_demo_cases.py: the rules that choose the demo cases, and the files it freezes. The rules decide what the demo claims to
show ("the safeguards catch this mistake", "they do not catch that one"), so each is tested against hand-made records.
"""
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import build_demo_cases as dc

ROOT = Path(__file__).resolve().parent.parent
_ids = iter(range(1, 10_000))


def rec(true="Pneumonia", pred="Pneumonia", p=0.9, status="agree", low=False, n=3, labs=True, alert=False, sets=None, pc=0.1, **kw):
    sid = kw.pop("study_id", next(_ids))
    return {"study_id": sid, "true": true, "pred": pred, "max_prob": p, "p_cancer": pc, "status": status, "n_inputs": n,
            "inputs": ["image", "text", "blood"][:n], "has_labs": labs, "low": low, "set": sets if sets is not None else [pred], "alert": alert,
            "case": {"image_path": None, "indication": "ind", "labs": {"wbc": 9.0, "crp": None}}}


def pick(profile_id, records):
    out = {c["profile"]["id"]: c["record"] for c in dc.select(records)}
    return out.get(profile_id)


def test_every_profile_has_an_id_a_title_a_question_a_note_and_the_ids_are_unique():
    ids = [p["id"] for p in dc.PROFILES]
    assert len(ids) == len(set(ids)) >= 9
    for p in dc.PROFILES:
        assert p["title"] and p["question"] and p["note"] and callable(p["pick"]) and callable(p["rank"])
    assert {"wrong-but-flagged", "wrong-and-confident", "missed-cancer-alert-fires"} <= set(ids)       # failures are shown on purpose
    titles = {p["id"]: p["title"] for p in dc.PROFILES}
    assert "alert catches" in titles["missed-cancer-alert-fires"] and "does not know" in titles["wrong-and-confident"]       # titles say what happens


def test_the_selection_is_deterministic_and_uses_each_study_once():
    records = [rec(true=t, pred=t, p=0.5 + i / 100) for i, t in enumerate(["Normal", "Pneumonia"] * 20)] + \
              [rec(true="Lung Cancer", pred="Lung Cancer", p=0.8, status="partial"), rec(true="Lung Cancer", pred="Lung Cancer", p=0.7, status="partial")]
    a, b = dc.select(records), dc.select(list(reversed(records)))
    assert [c["record"]["study_id"] for c in a] == [c["record"]["study_id"] for c in b]
    ids = [c["record"]["study_id"] for c in a]
    assert len(ids) == len(set(ids))


def test_a_clear_case_is_the_most_confident_one_that_meets_all_conditions():
    records = [rec("Normal", "Normal", p=0.80), rec("Normal", "Normal", p=0.95), rec("Normal", "Normal", p=0.97, low=True),
               rec("Normal", "Normal", p=0.99, status="partial"), rec("Normal", "Normal", p=0.99, n=2), rec("Normal", "Pneumonia", p=0.99)]
    chosen = pick("clear-normal", records)
    assert chosen["max_prob"] == 0.95 and chosen["n_inputs"] == 3 and chosen["status"] == "agree" and not chosen["low"]


def test_the_cancer_the_system_finds_must_be_a_confident_three_input_cancer():
    records = [rec("Lung Cancer", "Lung Cancer", p=0.9, low=True), rec("Lung Cancer", "Lung Cancer", p=0.8, n=2),
               rec("Lung Cancer", "Lung Cancer", p=0.7), rec("Normal", "Lung Cancer", p=0.99)]
    assert pick("clear-lung-cancer", records)["max_prob"] == 0.7


def test_the_disagreement_case_is_one_where_the_pooled_answer_is_right():
    records = [rec(status="conflict", p=0.6, true="Pneumonia", pred="Normal"), rec(status="conflict", p=0.5, true="Pneumonia", pred="Pneumonia"),
               rec(status="conflict", p=0.9, n=2)]
    assert pick("models-disagree-resolved", records)["max_prob"] == 0.5


def test_the_low_confidence_case_is_the_one_nearest_a_toss_up():
    records = [rec(low=True, p=0.40), rec(low=True, p=0.46), rec(low=True, p=0.55), rec(low=False, p=0.47)]
    assert pick("low-confidence", records)["max_prob"] == 0.46


def test_the_no_labs_case_has_no_labs_and_is_right_and_not_flagged():
    records = [rec(labs=False, n=2, p=0.8), rec(labs=False, n=2, p=0.9, low=True), rec(labs=False, n=2, true="Normal", pred="Pneumonia", p=0.95),
               rec(labs=True, n=2, p=0.99)]            # two inputs, so no earlier profile claims it first
    assert pick("no-labs", records)["max_prob"] == 0.8


def test_a_wrong_but_flagged_case_is_wrong_flagged_and_still_covered_by_its_prediction_set():
    # two-input cases, so the earlier "low-confidence" profile (which wants three) does not claim them first: each study is used once
    covered = rec(true="Lung Cancer", pred="Normal", low=True, p=0.4, sets=["Normal", "Lung Cancer"], n=2)
    uncovered = rec(true="Lung Cancer", pred="Normal", low=True, p=0.3, sets=["Normal"], n=2)
    unflagged = rec(true="Lung Cancer", pred="Normal", low=False, p=0.3, sets=["Normal", "Lung Cancer"], n=2)
    right = rec(true="Normal", pred="Normal", low=True, p=0.3, sets=["Normal"], n=2)
    assert pick("wrong-but-flagged", [uncovered, unflagged, right]) is None
    assert pick("wrong-but-flagged", [uncovered, covered, unflagged, right]) is covered


def test_the_alert_case_is_a_true_cancer_the_headline_misses_while_the_alert_fires():
    hit = rec(true="Lung Cancer", pred="Pneumonia", alert=True, pc=0.35)
    others = [rec(true="Lung Cancer", pred="Pneumonia", alert=False, pc=0.5), rec(true="Lung Cancer", pred="Lung Cancer", alert=True),
              rec(true="Normal", pred="Pneumonia", alert=True, pc=0.9)]
    assert pick("missed-cancer-alert-fires", others) is None
    assert pick("missed-cancer-alert-fires", others + [hit]) is hit


def test_the_honest_failure_is_wrong_and_unflagged_and_all_inputs_agree():
    bad = rec(true="Pneumonia", pred="Normal", p=0.9)
    flagged = rec(true="Pneumonia", pred="Normal", low=True, n=2)                # two inputs, so no earlier profile claims these first
    disagree = rec(true="Pneumonia", pred="Normal", status="conflict", n=2)
    assert pick("wrong-and-confident", [flagged, disagree, rec(true="Pneumonia", pred="Pneumonia", n=2)]) is None
    assert pick("wrong-and-confident", [bad]) is bad


def test_a_study_that_fits_two_profiles_is_shown_once_and_the_second_profile_takes_the_next_best():
    both = rec(true="Lung Cancer", pred="Normal", low=True, p=0.47, sets=["Normal", "Lung Cancer"])      # fits "low-confidence" and "wrong-but-flagged"
    other = rec(true="Lung Cancer", pred="Normal", low=True, p=0.30, sets=["Normal", "Lung Cancer"])
    chosen = {c["profile"]["id"]: c["record"]["study_id"] for c in dc.select([both, other])}
    assert chosen["low-confidence"] == both["study_id"] and chosen["wrong-but-flagged"] == other["study_id"]
    assert len(set(chosen.values())) == len(chosen)
    only = {c["profile"]["id"] for c in dc.select([both])}
    assert only == {"low-confidence"}                                         # the second profile gets nothing rather than a repeat


def test_a_profile_with_no_candidate_is_skipped_not_faked():
    chosen = dc.select([rec("Normal", "Normal", p=0.9)])
    assert [c["profile"]["id"] for c in chosen] == ["clear-normal"]


def test_freezing_copies_the_image_writes_the_manifest_and_keeps_the_answer_for_the_presenter(tmp_path):
    img = tmp_path / "src.jpg"
    Image.new("L", (64, 64), 100).save(img, "JPEG")
    r = rec("Normal", "Normal", p=0.9)
    r["case"]["image_path"] = str(img)
    out = tmp_path / "demo"
    manifest = dc.freeze(dc.select([r]), out)
    entry = manifest["cases"][0]
    assert (out / "clear-normal" / "image.jpg").is_file() and json.loads((out / "manifest.json").read_text())["cases"][0]["id"] == "clear-normal"
    assert entry["labs"] == {"wbc": 9.0}                                      # a lab that was not measured is not sent as a value
    assert entry["expected"]["true_label"] == "Normal" and entry["study_id"] == r["study_id"]
    assert entry["expected"]["consistency"] == "agree" and entry["candidates_for_this_profile"] == 1


def test_the_script_never_names_the_sealed_split():
    code = (ROOT / "scripts" / "build_demo_cases.py").read_text(encoding="utf-8")
    assert "test.csv" not in code and '"test"' not in code and "'test'" not in code
    assert '"val"' in code


@pytest.mark.slow
def test_the_frozen_demo_cases_reproduce_under_the_live_system():
    manifest = ROOT / "demo" / "cases" / "manifest.json"
    if not manifest.exists():
        pytest.skip("demo cases have not been built")
    from app import xray_guard
    from app.pipelines.staged import provided_inputs, stage_predict
    for c in json.loads(manifest.read_text())["cases"]:
        image = ROOT / "demo" / "cases" / c["id"] / "image.jpg"
        assert xray_guard.check(Image.open(image))[0], c["id"]
        pred = stage_predict({"deps": {}, "inputs": provided_inputs(c["indication"], c["labs"] or None, str(image))})
        assert pred["belief"]["predicted_class"] == c["expected"]["predicted_class"], c["id"]
        assert pred["decision"]["low_confidence"] == c["expected"]["low_confidence"], c["id"]
