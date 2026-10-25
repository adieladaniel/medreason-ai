"""Evidence assembly, writer, faithfulness check and the end-to-end invariant that RAG cannot change the label.
The retriever, the Ollama call and the NLI model are stubbed at their seams, so these tests need no models."""
import pytest

from app.rag import evidence, explain_case, faithfulness, writer
from tests.conftest import make_agent


def hit(cid, text="Low albumin was associated with mortality in pneumonia.", score=9.0):
    return {"chunk_id": cid, "text": text, "heading": "Results", "citation": f"Author {cid} 2024",
            "title": f"Paper {cid}", "url": f"https://example.org/{cid}", "rerank_score": score}


def q(text, driver, terms=("albumin",), modality="blood"):
    return {"text": text, "modality": modality, "driver": driver, "effect": 0.3, "terms": terms}


@pytest.fixture
def stub_retrieve(monkeypatch):
    import app.rag.retrieve as r

    def install(table):
        monkeypatch.setattr(r, "retrieve", lambda query, k=5: table.get(query, [])[:k])
    return install


# ------------------------------------------------------------------ evidence
def test_assemble_numbers_dedupes_and_records_every_driver(stub_retrieve):
    stub_retrieve({"q1": [hit(7), hit(8)], "q2": [hit(7)]})
    ev = evidence.assemble({"queries": [q("q1", "albumin = 2.8"), q("q2", "LDH = 400", ("ldh",))]})
    assert [p["n"] for p in ev["passages"]] == [1, 2]
    p7 = ev["passages"][0]
    assert p7["chunk_id"] == 7 and p7["drivers"] == ["albumin = 2.8", "LDH = 400"]
    assert p7["terms"] == {"albumin", "ldh"}


def test_assemble_reports_queries_with_no_passage(stub_retrieve):
    stub_retrieve({"q1": []})
    ev = evidence.assemble({"queries": [q("q1", "WBC = 31")]})
    assert ev["passages"] == [] and [u["driver"] for u in ev["unanswered"]] == ["WBC = 31"]


def test_assemble_caps_total_passages(stub_retrieve):
    stub_retrieve({f"q{i}": [hit(i * 10), hit(i * 10 + 1)] for i in range(5)})
    ev = evidence.assemble({"queries": [q(f"q{i}", f"d{i}") for i in range(5)]})
    assert len(ev["passages"]) == evidence.MAX_PASSAGES


def test_an_ungated_query_ungates_its_passage(stub_retrieve):
    stub_retrieve({"q1": [hit(1)], "diff": [hit(1)]})
    ev = evidence.assemble({"queries": [q("q1", "albumin"), q("diff", "disagree", None, "agent")]})
    assert ev["passages"][0]["terms"] is None


# ------------------------------------------------------------------ writer
def passage(n=1, terms=("albumin",), drivers=("albumin = 2.8",)):
    return {"n": n, "chunk_id": n, "text": "Low albumin was associated with mortality.", "heading": "Results",
            "title": "T", "citation": "C", "url": "u", "drivers": list(drivers), "queries": ["pneumonia with low albumin"],
            "terms": None if terms is None else set(terms), "rerank_score": 9.0}


def test_no_passages_means_the_model_is_never_called(monkeypatch, pneumonia_agent):
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: pytest.fail("the LLM must not be called without evidence"))
    out = writer.write_explanation(pneumonia_agent, {"queries": [q("x", "albumin = 2.8")]}, {"passages": [], "unanswered": []})
    assert out["background"] == [] and out["no_evidence_for"] == ["albumin = 2.8"]


def test_not_addressed_and_off_topic_sentences_are_dropped(monkeypatch, pneumonia_agent):
    replies = iter(["NOT_ADDRESSED", "A follow-up scan showed persistent consolidation.", "Low albumin is associated with mortality."])
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: next(replies))
    ev = {"passages": [passage(1), passage(2), passage(3)], "unanswered": []}
    plan = {"queries": [q("pneumonia with low albumin", "albumin = 2.8")]}
    out = writer.write_explanation(pneumonia_agent, plan, ev)
    assert [b["n"] for b in out["background"]] == [3]
    assert out["not_addressed"] == [1, 2]


def test_driver_without_a_kept_sentence_is_listed_as_missing(monkeypatch, pneumonia_agent):
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: "Something about lymphocytes.")
    ev = {"passages": [passage(1, drivers=("WBC = 31.0",), terms=("white blood",))], "unanswered": []}
    out = writer.write_explanation(pneumonia_agent, {"queries": [q("x", "WBC = 31.0", ("white blood",))]}, ev)
    assert out["background"] == [] and out["no_evidence_for"] == ["WBC = 31.0"]


def test_case_sentences_come_from_the_agent_facts(pneumonia_agent):
    s = writer.case_sentences(pneumonia_agent, {"queries": [q("x", "WBC = 31.0")]})
    assert s[0].startswith("The trained models predict Pneumonia with 90% probability")
    assert "All three agree." in s[1] and "WBC = 31.0 (blood)" in s[2]


# ------------------------------------------------------------------ faithfulness (NLI stubbed)
@pytest.mark.parametrize("text,expected", [
    ("associated with all-cause mortality", []),
    ("cause-specific survival", []),
    ("low albumin causes death", ["causes"]),
    ("this proves the diagnosis", ["proves"]),
    ("patients should receive antibiotics", ["should"]),
    ("the treatment arm did better", ["treatment"]),
])
def test_banned_wording(text, expected):
    assert sorted({m.group(0).lower() for m in faithfulness.BANNED.finditer(text)}) == expected


def test_numbers_must_appear_in_the_passage():
    assert faithfulness._numbers("mortality 5% to 17%, 10^9 cells") == {"5", "17", "109"}
    assert faithfulness._numbers("5-year survival under 20%") == {"5", "20"}


def test_check_sentence_applies_all_three_tests(monkeypatch):
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 0.9)
    ok = faithfulness.check_sentence("Albumin was lower in 20% of cases.", "Albumin was lower in 20% of cases.")
    assert ok["passed"]
    bad_num = faithfulness.check_sentence("Albumin was lower in 40% of cases.", "Albumin was lower in 20% of cases.")
    assert not bad_num["passed"] and bad_num["numbers_not_in_passage"] == ["40"]
    assert not faithfulness.check_sentence("This causes death.", "x")["passed"]
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 0.1)
    assert not faithfulness.check_sentence("Albumin matters.", "Albumin matters.")["passed"]


def test_verify_drops_failures_and_citations_to_missing_passages(monkeypatch):
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 0.9 if "good" in s else 0.0)
    expl = {"background": [{"n": 1, "text": "good sentence"}, {"n": 1, "text": "bad sentence"},
                           {"n": 9, "text": "good sentence about a passage that does not exist"}]}
    out = faithfulness.verify(expl, {"passages": [passage(1)]})
    assert [b["text"] for b in out["verified"]] == ["good sentence"]
    assert len(out["dropped"]) == 2 and out["pass_rate"] == pytest.approx(1 / 3)
    assert "does not exist" in out["dropped"][1]["reasons"][0]


# ------------------------------------------------------------------ the design rule
def test_the_label_cannot_be_changed_by_anything_the_writer_says(monkeypatch, pneumonia_agent, stub_retrieve):
    """RAG explains, it never diagnoses: even a writer that announces another diagnosis leaves the label alone."""
    stub_retrieve({"pneumonia with low albumin": [hit(1)], "pneumonia with elevated white blood cell count": [hit(2)]})
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: "The correct diagnosis is Lung Cancer, not low albumin.")
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 1.0)
    out = explain_case.explain(pneumonia_agent)
    assert out["predicted_class"] == "Pneumonia"
    assert pneumonia_agent["belief"]["predicted_class"] == "Pneumonia"


def test_only_verified_sentences_are_shown_with_their_reference(monkeypatch, pneumonia_agent, stub_retrieve):
    stub_retrieve({"pneumonia with low albumin": [hit(1)]})
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: "Low albumin is associated with mortality.")
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 0.0)       # checker rejects everything
    out = explain_case.explain(pneumonia_agent)
    assert out["references"] == [] and out["pass_rate"] == 0.0
    assert not any("[" in line and "]" in line for line in out["text"][3:] if "removed" not in line)


# ------------------------------------------------------------------ fallback query for image findings
def test_image_queries_carry_a_relational_fallback(pneumonia_agent):
    from app.rag.planner import plan_queries
    img = [q for q in plan_queries(pneumonia_agent)["queries"] if q["modality"] == "image"]
    assert img and all(q["fallback"].endswith("versus pneumonia chest radiograph") for q in img)


def test_fallback_is_used_only_when_the_primary_query_finds_nothing(stub_retrieve):
    stub_retrieve({"primary": [], "fallback": [hit(5, "Pulmonary edema can mimic pneumonia.")]})
    ev = evidence.assemble({"queries": [{**q("primary", "Edema = 0.5", ("edema",), "image"), "fallback": "fallback"}]})
    assert [p["chunk_id"] for p in ev["passages"]] == [5] and ev["passages"][0]["queries"] == ["fallback"]
    assert ev["unanswered"] == []


def test_fallback_is_skipped_when_the_primary_query_succeeds(stub_retrieve):
    stub_retrieve({"primary": [hit(1)], "fallback": [hit(2)]})
    ev = evidence.assemble({"queries": [{**q("primary", "Lung Opacity = 0.7", ("albumin",), "image"), "fallback": "fallback"}]})
    assert [p["chunk_id"] for p in ev["passages"]] == [1]


def test_query_with_no_primary_and_no_fallback_hits_is_reported_unanswered(stub_retrieve):
    stub_retrieve({})
    ev = evidence.assemble({"queries": [{**q("primary", "Edema = 0.5", ("edema",), "image"), "fallback": "fallback"}]})
    assert ev["passages"] == [] and len(ev["unanswered"]) == 1


# ------------------------------------------------------------------ recording the model calls
def test_every_model_call_goes_to_the_recorder_with_the_exact_prompt(monkeypatch, pneumonia_agent):
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: "Low albumin is associated with mortality.")
    calls = []
    ev = {"passages": [passage(1), passage(2)], "unanswered": []}
    writer.write_explanation(pneumonia_agent, {"queries": [q("pneumonia with low albumin", "albumin = 2.8")]}, ev,
                             model="m1", recorder=lambda **c: calls.append(c))
    assert [c["purpose"] for c in calls] == ["background_sentence"] * 2 and {c["model"] for c in calls} == {"m1"}
    system, user = writer.build_prompt(ev["passages"][0], "pneumonia with low albumin")
    assert calls[0]["prompt"] == f"{system}\n\n{user}" and "Low albumin was associated with mortality." in calls[0]["prompt"]
    assert calls[0]["response"] == "Low albumin is associated with mortality." and calls[0]["seconds"] >= 0


def test_compose_gives_the_same_text_as_explain(monkeypatch, pneumonia_agent, stub_retrieve):
    """explain() is now plan, assemble, write, verify, compose; the staged pipeline runs the same steps one by one."""
    from app.rag.planner import plan_queries
    stub_retrieve({"pneumonia with low albumin": [hit(1)]})
    monkeypatch.setattr(writer, "_ask", lambda *a, **k: "Low albumin is associated with mortality.")
    monkeypatch.setattr(faithfulness, "entailment", lambda s, p: 0.9)
    whole = explain_case.explain(pneumonia_agent)
    plan = plan_queries(pneumonia_agent)
    ev = evidence.assemble(plan)
    stepwise = explain_case.compose(plan, ev, faithfulness.verify(writer.write_explanation(pneumonia_agent, plan, ev), ev))
    assert stepwise == whole
