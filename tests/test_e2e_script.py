"""
The logic of scripts/run_e2e.py that does not need a server: how cases are chosen, what the offline answer for a case is,
how results are summarised, and that a case is always deleted from the server after it is read (it holds a real X-ray).
"""
import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import run_e2e as e2e

LABS = ["wbc", "hemoglobin", "albumin"]
CLASSES = e2e.CLASSES


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, n, p=[0.3, 0.6, 0.1])
    wrong = rng.random(n) < 0.35
    fused = np.where(wrong, rng.choice(CLASSES, n), label)
    df = pd.DataFrame({"study_id": np.arange(1000, 1000 + n), "final_label": label, "fused_pred": fused})
    for c in LABS:
        df[c] = np.where(rng.random(n) < 0.15, np.nan, rng.random(n) * 10)
    df.loc[rng.random(n) < 0.1, LABS] = np.nan                                  # some studies with no labs at all
    return df


# ------------------------------------------------------------------ choosing cases
def test_selection_is_repeatable_and_the_seed_changes_it():
    df = frame()
    a, b = e2e.select_cases(df, LABS, 42), e2e.select_cases(df, LABS, 42)
    assert a == b
    assert [c["study_id"] for c in e2e.select_cases(df, LABS, 7)] != [c["study_id"] for c in a]


def test_selection_covers_every_category_once_per_study():
    chosen = e2e.select_cases(frame(), LABS, 42)
    assert len(chosen) == 16
    assert len({c["study_id"] for c in chosen}) == 16
    assert {c["category"] for c in chosen} == {"agree, pneumonia", "agree, normal", "cancer found by fusion", "cancer missed by fusion",
                                              "wrong, not cancer", "no labs on record", "text only", "image only", "no X-ray",
                                              "no indication"}
    drops = {c["category"]: c["drop"] for c in chosen}
    assert drops["text only"] == ["image", "blood"] and drops["image only"] == ["blood", "text"]
    assert drops["no X-ray"] == ["image"] and drops["no indication"] == ["text"] and drops["agree, normal"] == []


def test_every_chosen_study_really_belongs_to_its_category():
    df = frame()
    by_id = df.set_index("study_id")
    for c in e2e.select_cases(df, LABS, 42):
        row = by_id.loc[c["study_id"]]
        has_labs = row[LABS].notna().any()
        right = row.final_label == row.fused_pred
        cat = c["category"]
        if cat == "no labs on record":
            assert not has_labs
            continue
        assert has_labs
        if cat == "agree, normal":
            assert row.final_label == "Normal" and right
        elif cat == "cancer found by fusion":
            assert row.final_label == "Lung Cancer" and right
        elif cat == "cancer missed by fusion":
            assert row.final_label == "Lung Cancer" and not right
        elif cat == "wrong, not cancer":
            assert row.final_label != "Lung Cancer" and not right
        else:                                                                    # pneumonia the fusion gets right
            assert row.final_label == "Pneumonia" and right


def test_a_category_with_too_few_studies_gives_what_exists_instead_of_failing():
    df = frame(60)
    df.loc[df.final_label == "Lung Cancer", "final_label"] = "Pneumonia"        # no cancers at all
    chosen = e2e.select_cases(df, LABS, 42)
    assert not [c for c in chosen if c["category"].startswith("cancer")]
    assert len(chosen) >= 8


# ------------------------------------------------------------------ the offline twin
def stored_frames(study_id=5):
    def one(prefix, p):
        return pd.DataFrame({"study_id": [study_id], **{f"{prefix}{k}": [v] for k, v in zip(("normal", "pneumonia", "lung_cancer"), p)}})
    return {"image": one("image_p_", (0.2, 0.7, 0.1)), "blood": one("blood_p_", (0.1, 0.8, 0.1)),
            "text": one("text_indication_p_", (0.05, 0.15, 0.80))}


def test_expected_answer_pools_only_the_inputs_that_were_sent():
    stored = stored_frames()
    assert e2e.expected_for({"text"}, 5, stored)["predicted_class"] == "Lung Cancer"
    assert e2e.expected_for({"image", "blood"}, 5, stored)["predicted_class"] == "Pneumonia"
    both = e2e.expected_for({"image", "blood", "text"}, 5, stored)
    assert both["predicted_class"] in CLASSES and pytest.approx(sum(both["probabilities"].values()), abs=1e-6) == 1.0
    assert both["probabilities"] != e2e.expected_for({"image", "blood"}, 5, stored)["probabilities"]


# ------------------------------------------------------------------ summarising
def rec(cat="agree, normal", status="done", wall=50.0, ttp=1.0, pred="Normal", exp="Normal", diff=0.0, correct=True,
        timings=None, verified=2, dropped=1, calls=(1.0, 2.0), passages=3):
    return {"category": cat, "status": status, "wall_seconds": wall, "time_to_prediction": ttp, "predicted_class": pred,
            "expected_class": exp, "class_matches_offline": pred == exp, "max_probability_difference": diff, "correct": correct,
            "timings": timings or {"predicting": 1.0, "explaining": 5.0, "retrieving": 7.0, "writing": 3.0, "verifying": 12.0},
            "explainers_match_stored_prediction": True, "n_passages": passages, "n_verified": verified, "n_dropped": dropped,
            "llm_call_seconds": list(calls)}


def test_summary_spreads_agreement_and_evidence_counts():
    rows = [rec(wall=w, timings={"predicting": 1.0, "explaining": 5.0, "retrieving": 7.0, "writing": 3.0, "verifying": v})
            for w, v in ((40, 6.0), (50, 12.0), (60, 18.0))]
    rows.append(rec(cat="cancer missed by fusion", pred="Pneumonia", exp="Lung Cancer", correct=False, diff=0.4))
    s = e2e.summarise(rows)
    assert s["cases_run"] == 4 and s["statuses"] == {"done": 4}
    assert s["stage_seconds"]["verifying"] == {"n": 4, "p50": 12.0, "p95": pytest.approx(17.1, abs=0.2), "max": 18.0}
    assert s["same_class_as_offline"] == {"n": 4, "same": 3}
    assert s["agrees_with_true_label_in_this_selected_set"] == {"n": 4, "agree": 3}
    assert s["max_probability_difference_vs_offline"]["max"] == 0.4
    assert s["evidence"] == {"cases_with_evidence": 4, "cases_with_a_verified_sentence": 4, "verified_sentences": 8, "dropped_sentences": 4}
    assert s["by_category"]["cancer missed by fusion"] == {"n": 1, "correct": 0, "matches_offline": 0}
    assert s["language_model_call_seconds"]["n"] == 8
    assert s["verifying_seconds_per_drafted_sentence"]["p50"] == pytest.approx(4.0, abs=0.01)       # 12 s over 3 sentences


def test_agreeing_with_the_offline_twin_is_not_the_same_as_being_right():
    rows = [rec(pred="Pneumonia", exp="Normal", correct=True),       # differs from offline, happens to be right
            rec(pred="Pneumonia", exp="Normal", correct=True),       # the same again
            rec(pred="Normal", exp="Normal", correct=False)]         # same as offline, wrong
    s = e2e.summarise(rows)
    assert s["same_class_as_offline"] == {"n": 3, "same": 1}
    assert s["agrees_with_true_label_in_this_selected_set"] == {"n": 3, "agree": 2}
    assert s["by_category"]["agree, normal"] == {"n": 3, "correct": 2, "matches_offline": 1}


def test_summary_survives_rejected_and_failed_cases_and_an_empty_run():
    rows = [rec(), {"category": "x", "status": "rejected", "http_status": 429}, rec(status="partial", timings={"predicting": 1.0}, verified=None, dropped=None)]
    rows[2].pop("n_verified"), rows[2].pop("n_dropped")
    s = e2e.summarise(rows)
    assert s["statuses"] == {"done": 1, "partial": 1, "rejected": 1}
    assert s["stage_seconds"]["predicting"]["n"] == 2 and s["stage_seconds"]["writing"]["n"] == 1
    empty = e2e.summarise([])
    assert empty["cases_run"] == 0 and empty["wall_seconds"] == {"n": 0}


# ------------------------------------------------------------------ talking to the server
class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.text = body, status, str(body)

    def json(self):
        return self._body


class FakeClient:
    def __init__(self, views, post_status=202, fail_on_poll=None):
        self.views, self.post_status, self.fail_on_poll = list(views), post_status, fail_on_poll
        self.deleted, self.posted, self.polls = [], None, 0

    def post(self, url, data=None, files=None, timeout=None):
        self.posted = {"data": data, "files": files}
        return FakeResponse({"id": "a" * 32}, self.post_status)

    def get(self, url, timeout=None):
        if url.endswith("/llm-calls"):
            return FakeResponse({"calls": [{"seconds": 1.5}, {"seconds": 0.5}]})
        self.polls += 1
        if self.fail_on_poll is not None and self.polls >= self.fail_on_poll:
            raise RuntimeError("connection lost")
        return FakeResponse(self.views.pop(0) if len(self.views) > 1 else self.views[0])

    def delete(self, url, timeout=None):
        self.deleted.append(url)
        return FakeResponse({}, 204)


def done_view():
    probs = {"Normal": 0.1, "Pneumonia": 0.8, "Lung Cancer": 0.1}
    return {"status": "done", "error": None, "timings": {"predicting": 1.0}, "prediction": {
        "belief": {"predicted_class": "Pneumonia", "probabilities": probs}, "consistency": {"status": "agree"},
        "decision": {"low_confidence": False, "prediction_set": ["Pneumonia"], "cancer_alert": {"flag": False}}},
        "explanations": {"text": {"matches_prediction": True}},
        "evidence": {"passages": [1, 2], "unanswered": [], "final": {"verified": [1], "dropped": [], "pass_rate": 1.0}}}


@pytest.fixture
def patched_case(monkeypatch, tmp_path):
    import app.agent.orchestrator as orch
    buf = io.BytesIO()
    Image.new("L", (80, 80), 90).save(buf, "PNG")
    img = tmp_path / "x.jpg"
    img.write_bytes(buf.getvalue())
    seen = []

    def fake_case(study_id, split, drop):
        seen.append(split)
        return {"true_label": "Pneumonia", "image_path": str(img), "labs": {"wbc": 12.0, "albumin": None}, "indication": "fever"}

    monkeypatch.setattr(orch, "_case_from_split", fake_case)
    return seen


def test_a_case_is_read_compared_with_its_offline_twin_and_deleted(patched_case):
    client = FakeClient([{"status": "running", "prediction": None}, done_view()])
    spec = {"study_id": 5, "category": "agree, pneumonia", "drop": []}
    r = e2e.run_one(client, "http://s", spec, stored_frames(), poll=0)
    assert patched_case == ["val"]                                              # only ever the validation split
    assert sorted(client.posted["data"]) == ["indication", "labs"] and "image" in client.posted["files"]
    assert r["inputs_sent"] == ["blood", "image", "text"] and r["status"] == "done"
    assert r["predicted_class"] == "Pneumonia" and r["expected_class"] in CLASSES and "class_matches_offline" in r
    assert r["n_passages"] == 2 and r["n_verified"] == 1 and r["llm_call_seconds"] == [1.5, 0.5]
    assert r["time_to_prediction"] is not None and r["wall_seconds"] >= r["time_to_prediction"]
    assert client.deleted == ["http://s/api/cases/" + "a" * 32]


def test_a_dropped_input_is_not_sent_and_not_used_for_the_offline_answer(patched_case):
    client = FakeClient([done_view()])
    spec = {"study_id": 5, "category": "text only", "drop": ["image", "blood"]}
    r = e2e.run_one(client, "http://s", spec, stored_frames(), poll=0)
    assert sorted(client.posted["data"]) == ["indication"] and client.posted["files"] is None
    assert r["inputs_sent"] == ["text"] and r["expected_class"] == "Lung Cancer"


def test_the_case_is_deleted_even_when_polling_fails(patched_case):
    client = FakeClient([{"status": "running", "prediction": None}], fail_on_poll=2)
    with pytest.raises(RuntimeError):
        e2e.run_one(client, "http://s", {"study_id": 5, "category": "c", "drop": []}, stored_frames(), poll=0)
    assert len(client.deleted) == 1


def test_a_refused_case_is_reported_and_nothing_is_deleted(patched_case):
    client = FakeClient([done_view()], post_status=429)
    r = e2e.run_one(client, "http://s", {"study_id": 5, "category": "c", "drop": []}, stored_frames(), poll=0)
    assert r["status"] == "rejected" and r["http_status"] == 429 and client.deleted == []
