"""The staged pipeline and its job runner, on a throwaway SQLite file with fake models.

The models, the retriever, the language model and the entailment checker are all replaced, so these
tests need no model files and no service. What they check is the pipeline's own behavior: what is
stored and when, what survives a failure, and that nothing after the prediction can change it."""
import numpy as np
import pytest

from app.database import repository as repo
from app.database.session import init_db, session_scope
from app.pipelines import runner, staged
from app.rag import faithfulness, writer
from tests.conftest import item, probs

PNEUMONIA = {"text": probs(0.05, 0.9, 0.05), "blood": probs(0.1, 0.8, 0.1), "image": probs(0.1, 0.8, 0.1)}
LABELS = ["Normal", "Pneumonia", "Lung Cancer"]


def prediction_of(modality, p):
    return {"modality": modality, "predicted_class": max(p, key=p.get), "probabilities": p}


def fake_predictors(table=PNEUMONIA):
    return {m: (lambda _inp, m=m: prediction_of(m, table[m])) for m in table}


ATTRIBUTIONS = {
    "blood": [item("WBC", 31.0, 0.5), item("albumin", 2.8, 0.3)],
    "image": [item("Lung Opacity", 0.8, 0.4)],
    "text": [item("fever", 1.0, 0.2)],
}


def fake_explainers(table=PNEUMONIA, calls=None, overrides=None):
    def make(m):
        def explain(inp):
            if calls is not None:
                calls.append(m)
            p = (overrides or {}).get(m, table[m])
            return {**prediction_of(m, p), "attributions": {"method": "fake", "items": ATTRIBUTIONS[m]},
                    "summary": f"{m} summary", "_cam": np.zeros((4, 4)), "_raw": {"big": np.zeros(3)}}
        return explain
    return {m: make(m) for m in table}


def hit(cid, text):
    return {"chunk_id": cid, "text": text, "heading": "Results", "citation": f"Author {cid} 2024",
            "title": f"Paper {cid}", "url": f"https://example.org/{cid}", "rerank_score": 9.0}


@pytest.fixture
def url(tmp_path):
    u = f"sqlite:///{(tmp_path / 'pipeline.db').as_posix()}"
    init_db(u)
    return u


@pytest.fixture
def evidence_stubs(monkeypatch):
    """A retriever with passages for two drivers, a writer that answers, and an entailment checker that agrees."""
    import app.rag.retrieve as r
    table = {"pneumonia with low albumin": [hit(1, "Low albumin was associated with mortality in pneumonia.")],
             "pneumonia with elevated white blood cell count": [hit(2, "Leukocytosis was common in pneumonia.")]}
    monkeypatch.setattr(r, "retrieve", lambda query, k=5: table.get(query, [])[:k])
    asked = []
    monkeypatch.setattr(writer, "_ask", lambda p, topic, model, timeout: asked.append(topic) or "Low albumin is associated with mortality in pneumonia.")
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 0.9)
    return asked


def new_case(url, **inputs):
    inputs = {"indication": "fever and cough", "labs": {"wbc": 31.0, "albumin": 2.8}, "image_path": "x.png", **inputs}
    with session_scope(url) as s:
        return repo.create_case(s, **inputs).id


def read(url, case_id):
    with session_scope(url) as s:
        c = repo.get_case(s, case_id)
        c.events, c.llm_calls                      # load the children before the session closes
        return c


def deps(tmp_path, **extra):
    figures = {m: (lambda res, path: path.write_bytes(b"png")) for m in ("blood", "image", "text")}
    return {"predictors": fake_predictors(), "explainers": fake_explainers(), "plotters": figures, **extra}


# ------------------------------------------------------------------ inputs
def test_blank_indication_and_empty_labs_count_as_missing():
    assert staged.provided_inputs("  ", {"wbc": None, "rbc": None}, None) == {}
    assert staged.provided_inputs("fever", {"wbc": 12.0, "rbc": None}, "a.png") == {
        "text": "fever", "blood": {"wbc": 12.0, "rbc": None}, "image": "a.png"}
    assert staged.provided_inputs(None, {}, "a.png") == {"image": "a.png"}
    assert staged.provided_inputs("???", None, None) == {} and staged.provided_inputs("...", None, "a.png") == {"image": "a.png"}
    assert staged.provided_inputs("65M", None, None) == {"text": "65M"}


# ------------------------------------------------------------------ the same belief as run_case
@pytest.mark.parametrize("modalities", [["text", "blood", "image"], ["blood", "text"], ["image"], ["text", "image"]])
def test_staged_prediction_gives_the_same_belief_as_run_case(monkeypatch, modalities):
    """The point of staging is speed, not a different answer: for the same model outputs the belief,
    the agreement check and the plan must be identical to what run_case produces."""
    from app.agent import orchestrator
    table = {m: PNEUMONIA[m] for m in modalities}
    monkeypatch.setattr(orchestrator, "TOOLS", fake_explainers(table))
    inputs = {"image_path": "x.png" if "image" in modalities else None,
              "labs": {"wbc": 1.0} if "blood" in modalities else None,
              "indication": "fever" if "text" in modalities else None}
    full = orchestrator.run_case(**inputs)
    ctx = {"deps": {"predictors": fake_predictors(table)},
           "inputs": staged.provided_inputs(inputs["indication"], inputs["labs"], inputs["image_path"])}
    fast = staged.stage_predict(ctx)
    assert fast["belief"] == full["belief"]
    assert fast["consistency"] == full["consistency"]
    assert fast["missing_info_plan"] == full["missing_info_plan"]


# ------------------------------------------------------------------ a whole job
def test_a_job_stores_every_stage_and_finishes(url, tmp_path, evidence_stubs):
    cid = new_case(url)
    assert runner.run_job(cid, url, deps=deps(tmp_path), figure_dir=tmp_path / "figs") == "done"
    c = read(url, cid)
    assert (c.status, c.stage, c.error) == ("done", "done", None)
    assert c.prediction["belief"]["predicted_class"] == "Pneumonia"
    assert c.prediction["modalities_used"] == ["text", "blood", "image"] and set(c.prediction["seconds"]) == {"text", "blood", "image"}
    assert set(c.explanations) == {"blood", "image", "text"}
    assert "_cam" not in c.explanations["image"] and "_raw" not in c.explanations["blood"]      # arrays are not stored
    assert c.evidence["plan"]["queries"] and c.evidence["written"]["background"] and c.evidence["final"]["references"]
    assert set(c.timings) == {"predicting", "explaining", "retrieving", "writing", "verifying"}
    assert [(e.stage, e.status) for e in c.events] == [(st, status) for st in c.timings for status in ("started", "finished")]


def test_explanation_figures_are_saved_under_the_case_folder(url, tmp_path, evidence_stubs):
    cid = new_case(url)
    runner.run_job(cid, url, deps=deps(tmp_path), figure_dir=tmp_path / "figs")
    c = read(url, cid)
    assert c.explanations["blood"]["figure"] == f"{cid}/blood.png"
    assert (tmp_path / "figs" / cid / "blood.png").read_bytes() == b"png"


def test_a_figure_that_fails_to_draw_does_not_fail_the_case(url, tmp_path, evidence_stubs):
    def broken(res, path):
        raise RuntimeError("no display")
    cid = new_case(url)
    assert runner.run_job(cid, url, deps=deps(tmp_path, plotters={"blood": broken}), figure_dir=tmp_path / "figs") == "done"
    e = read(url, cid).explanations["blood"]
    assert e["figure"] is None and "no display" in e["figure_error"]


def test_every_model_call_is_stored_with_its_prompt_and_answer(url, tmp_path, evidence_stubs):
    cid = new_case(url)
    runner.run_job(cid, url, deps=deps(tmp_path), figure_dir=tmp_path)
    calls = read(url, cid).llm_calls
    assert len(calls) == 2 and all(c.purpose == "background_sentence" and c.model == "llama3.2:latest" for c in calls)
    by_passage = {("Leukocytosis" in c.prompt): c for c in calls}               # one call per retrieved passage
    assert set(by_passage) == {True, False}
    albumin = by_passage[False]
    assert "Low albumin was associated with mortality" in albumin.prompt and "summarise one passage" in albumin.prompt
    assert albumin.response.startswith("Low albumin") and albumin.seconds >= 0


def test_a_missing_modality_is_left_out_everywhere(url, tmp_path, evidence_stubs):
    cid = new_case(url, indication=None)
    runner.run_job(cid, url, deps=deps(tmp_path), figure_dir=tmp_path)
    c = read(url, cid)
    assert c.prediction["modalities_used"] == ["blood", "image"] and set(c.explanations) == {"blood", "image"}


# ------------------------------------------------------------------ streaming
def test_blood_and_image_explanations_are_readable_while_text_is_still_running(url, tmp_path, evidence_stubs):
    seen = []
    explainers = fake_explainers()
    slow_text = explainers["text"]

    def text_that_looks_at_the_database(inp):
        seen.append(sorted((read(url, cid).explanations or {}).keys()))        # what a client would see right now
        return slow_text(inp)
    explainers["text"] = text_that_looks_at_the_database
    cid = new_case(url)
    runner.run_job(cid, url, deps=deps(tmp_path, explainers=explainers), figure_dir=tmp_path)
    assert seen == [["blood", "image"]]


def test_the_prediction_is_readable_while_explanations_are_still_running(url, tmp_path, evidence_stubs):
    seen = []
    explainers = fake_explainers()
    first = explainers["blood"]

    def blood_that_looks_at_the_database(inp):
        c = read(url, cid)
        seen.append((c.stage, c.status, c.prediction["belief"]["predicted_class"]))
        return first(inp)
    explainers["blood"] = blood_that_looks_at_the_database
    cid = new_case(url)
    runner.run_job(cid, url, deps=deps(tmp_path, explainers=explainers), figure_dir=tmp_path)
    assert seen == [("explaining", "running", "Pneumonia")]


# ------------------------------------------------------------------ failures
def test_a_failed_explainer_leaves_the_prediction_and_earlier_explanations(url, tmp_path, evidence_stubs):
    explainers = fake_explainers()
    explainers["text"] = lambda inp: (_ for _ in ()).throw(RuntimeError("SHAP ran out of memory"))
    cid = new_case(url)
    assert runner.run_job(cid, url, deps=deps(tmp_path, explainers=explainers), figure_dir=tmp_path) == "partial"
    c = read(url, cid)
    assert c.status == "partial" and "explaining failed" in c.error and "SHAP ran out of memory" in c.error
    assert c.prediction["belief"]["predicted_class"] == "Pneumonia"             # the answer is still there
    assert set(c.explanations) == {"blood", "image"} and c.evidence is None      # later stages never ran
    assert (c.events[-1].stage, c.events[-1].status) == ("explaining", "failed")


def test_an_unreachable_language_model_keeps_the_explanations(url, tmp_path, evidence_stubs, monkeypatch):
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError("ollama is not running")))
    cid = new_case(url)
    assert runner.run_job(cid, url, deps=deps(tmp_path), figure_dir=tmp_path) == "partial"
    c = read(url, cid)
    assert "writing failed" in c.error and c.explanations and c.evidence["plan"] and "written" not in c.evidence


def test_a_failed_prediction_means_the_case_failed(url, tmp_path, evidence_stubs):
    predictors = fake_predictors()
    predictors["blood"] = lambda inp: (_ for _ in ()).throw(ValueError("bad lab value"))
    cid = new_case(url)
    assert runner.run_job(cid, url, deps=deps(tmp_path, predictors=predictors), figure_dir=tmp_path) == "failed"
    c = read(url, cid)
    assert c.status == "failed" and "predicting failed" in c.error and c.prediction is None and c.explanations is None


LEAKY = [
    FileNotFoundError(2, "No such file or directory", "/app/experiments/blood_baseline/model.joblib"),
    OSError("cannot open C:\\Users\\HP\\models\\text.joblib now"),
    OSError("cannot open C:/Users/HP/models/text.joblib now"),
    RuntimeError("HTTPConnectionPool: http://host.docker.internal:11434/api/chat failed"),
    RuntimeError("could not connect to postgresql+psycopg://medreason:secret@postgres:5432/medreason"),
]


@pytest.mark.parametrize("exc", LEAKY, ids=["posix path", "windows path", "windows path, slashes", "url with host", "database url"])
def test_a_failure_shown_to_a_client_carries_no_path_url_or_credential(url, tmp_path, evidence_stubs, exc, caplog):
    predictors = fake_predictors()
    predictors["blood"] = lambda inp: (_ for _ in ()).throw(exc)
    cid = new_case(url)
    with caplog.at_level("ERROR", logger="medreason.runner"):
        assert runner.run_job(cid, url, deps=deps(tmp_path, predictors=predictors), figure_dir=tmp_path) == "failed"
    c = read(url, cid)
    shown = c.error + " ".join(str(e.detail) for e in c.events)
    for secret in ("/app/", "experiments", "Users", "HP", "host.docker.internal", "11434", "secret", "postgres:5432", "model.joblib"):
        assert secret not in shown, (secret, shown)
    assert type(exc).__name__ in c.error                                   # still says what kind of failure it was
    assert str(exc.args[-1])[:12] in caplog.text or str(exc)[:30] in caplog.text   # the log keeps the full text for whoever runs the service


def test_an_ordinary_failure_message_is_kept_whole_and_a_long_one_is_cut():
    assert runner.public_message(RuntimeError("SHAP ran out of memory")) == "RuntimeError: SHAP ran out of memory"
    assert runner.public_message(ValueError("ratio 3/4 and/or 5/6, TCP/IP")) == "ValueError: ratio 3/4 and/or 5/6, TCP/IP"
    cut = runner.public_message(RuntimeError("x" * 1000))
    assert cut == "RuntimeError: " + "x" * runner.MAX_ERROR_CHARS


def test_a_case_with_nothing_usable_fails_without_running_a_model(url, tmp_path):
    cid = new_case(url, indication="   ", labs=None, image_path=None)
    boom = {"predictors": {m: (lambda inp: pytest.fail("no model should run")) for m in PNEUMONIA}}
    assert runner.run_job(cid, url, deps=boom, figure_dir=tmp_path) == "failed"
    assert "no usable input" in read(url, cid).error


# ------------------------------------------------------------------ the design rule
def test_nothing_after_the_prediction_can_change_it(url, tmp_path, evidence_stubs, monkeypatch):
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: "The correct diagnosis is Lung Cancer, with low albumin.")
    # the explainers even disagree with the stored prediction; it is recorded, not applied
    drift = {"blood": probs(0.1, 0.1, 0.8)}
    cid = new_case(url)
    runner.run_job(cid, url, deps=deps(tmp_path, explainers=fake_explainers(overrides=drift)), figure_dir=tmp_path)
    c = read(url, cid)
    assert c.prediction["belief"]["predicted_class"] == "Pneumonia"
    assert c.evidence["written"]["predicted_class"] == "Pneumonia"
    assert c.explanations["blood"]["matches_prediction"] is False and c.explanations["blood"]["max_probability_difference"] > 0.5
    assert c.explanations["image"]["matches_prediction"] is True


def test_a_normal_prediction_asks_for_no_literature_and_never_calls_the_model(url, tmp_path, monkeypatch):
    normal = {m: probs(0.9, 0.05, 0.05) for m in PNEUMONIA}
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: pytest.fail("the model must not be called without evidence"))
    cid = new_case(url)
    d = deps(tmp_path, predictors=fake_predictors(normal), explainers=fake_explainers(normal))
    assert runner.run_job(cid, url, deps=d, figure_dir=tmp_path) == "done"
    c = read(url, cid)
    assert c.evidence["plan"]["queries"] == [] and c.evidence["written"]["background"] == [] and c.llm_calls == []


# ------------------------------------------------------------------ the worker
def test_the_worker_runs_submitted_cases_in_order_and_survives_a_bad_one(url):
    ran = []
    stage = staged.Stage("predicting", "prediction", lambda ctx: ran.append(ctx["case_id"]) or {"ok": True})
    a, b = new_case(url), new_case(url)
    w = runner.JobRunner(url, stages=[stage], deps={"no_figures": True})
    w.start()
    w.submit("does-not-exist")               # run_job raises for this one; the worker must carry on
    w.submit(a)
    w.submit(b)
    w.stop(timeout=30)
    assert ran == [a, b]
    assert read(url, a).status == "done" and read(url, b).status == "done"
    assert w.pending() == 0


# ------------------------------------------------------------------ how sure the system is
def test_the_prediction_stage_adds_how_sure_it_is_and_leaves_the_belief_alone():
    from app.agent.reasoning import reason
    ctx = {"deps": {"predictors": fake_predictors()}, "inputs": {"text": "fever", "blood": {"wbc": 1.0}, "image": "x.png"}}
    out = staged.stage_predict(ctx)
    plain = reason({m: PNEUMONIA[m] for m in PNEUMONIA})
    assert out["belief"] == plain["belief"] and out["consistency"] == plain["consistency"]       # exactly what it was before
    d = out["decision"]
    assert d["prediction_set"] == ["Pneumonia"] and d["low_confidence"] is False and d["cancer_alert"]["flag"] is False


def test_the_rules_follow_the_inputs_that_actually_ran():
    out = staged.stage_predict({"deps": {"predictors": fake_predictors({m: PNEUMONIA[m] for m in ("text", "image")})},
                                "inputs": {"text": "fever", "image": "x.png"}})
    assert out["decision"]["confidence"]["threshold"] != 0.5535 and out["modalities_used"] == ["text", "image"]


def test_without_the_rules_file_the_prediction_is_still_made(monkeypatch, tmp_path):
    from app.agent import decision
    monkeypatch.setattr(decision, "RULES_PATH", tmp_path / "not_there.json")
    out = staged.stage_predict({"deps": {"predictors": fake_predictors()}, "inputs": {"text": "fever", "blood": {"wbc": 1.0}, "image": "x.png"}})
    assert out["decision"] is None and out["belief"]["predicted_class"] == "Pneumonia"


def test_the_stored_case_carries_the_decision(url, tmp_path, evidence_stubs):
    cid = new_case(url)
    runner.run_job(cid, url, deps=deps(tmp_path), figure_dir=tmp_path)
    assert read(url, cid).prediction["decision"]["prediction_set"] == ["Pneumonia"]
