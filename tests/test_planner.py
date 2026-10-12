"""Evidence planner: which drivers become queries, and which never do."""
import copy

from app.rag import planner
from app.rag.planner import BLOOD_RANGES, IMAGE_PHRASES, MAX_QUERIES, TERMS, plan_queries
from tests.conftest import item, make_agent, probs, tool


def texts(plan):
    return [q["text"] for q in plan["queries"]]


def test_normal_prediction_asks_for_nothing():
    plan = plan_queries(make_agent("Normal"))
    assert plan["queries"] == [] and "no literature" in plan["note"]


def test_abnormal_supporting_labs_become_queries_and_normal_ones_are_skipped(pneumonia_agent):
    plan = plan_queries(pneumonia_agent)
    assert "pneumonia with elevated white blood cell count" in texts(plan)
    assert "pneumonia with low albumin" in texts(plan)
    assert any("monocytes %" in s for s in plan["skipped"])          # supported, but within range


def test_opposing_features_never_seed_queries(pneumonia_agent):
    assert not any("hemoglobin" in q["driver"] for q in plan_queries(pneumonia_agent)["queries"])


def test_image_channels_get_one_query_each(pneumonia_agent):
    img = [q for q in plan_queries(pneumonia_agent)["queries"] if q["modality"] == "image"]
    assert [q["text"] for q in img] == ["pneumonia chest X-ray with lung opacity",
                                        "pneumonia chest X-ray with consolidation"]


def test_weak_and_unmapped_image_channels_are_ignored(pneumonia_agent):
    drivers = " ".join(q["driver"] for q in plan_queries(pneumonia_agent)["queries"])
    assert "Edema" not in drivers       # value 0.2 is below IMAGE_MIN_VALUE
    assert "Fracture" not in drivers    # strong, but no literature phrase for it


def test_text_filler_and_self_label_words_are_not_queried(pneumonia_agent):
    q = [q for q in plan_queries(pneumonia_agent)["queries"] if q["modality"] == "text"][0]
    assert q["text"] == "pneumonia with fever and cough"      # not "the", not "pneumonia" itself


def test_modality_that_disagrees_with_the_final_label_is_not_used():
    p = probs(0.2, 0.5, 0.3)
    agent = make_agent("Pneumonia", status="conflict", tools={
        "blood": tool("Normal", probs(0.8, 0.1, 0.1), [item("WBC", 31.0, 0.5)]),
        "text": tool("Lung Cancer", probs(0.1, 0.2, 0.7), [item("mass", 1.0, 0.4)]),
        "image": tool("Pneumonia", p, [item("Lung Opacity", 0.8, 0.4)])})
    plan = plan_queries(agent)
    assert not any(q["modality"] in ("blood", "text") for q in plan["queries"])
    assert any("blood model predicted Normal" in s for s in plan["skipped"])


def test_differential_query_names_the_disease_a_modality_predicted_and_comes_first():
    agent = make_agent("Pneumonia", status="conflict", tools={
        "text": tool("Lung Cancer", probs(0.1, 0.2, 0.7)),
        "image": tool("Pneumonia", probs(0.1, 0.8, 0.1), [item("Lung Opacity", 0.8, 0.4)])})
    plan = plan_queries(agent)
    assert plan["queries"][0]["text"].startswith("pneumonia versus lung cancer")
    assert plan["queries"][0]["terms"] is None       # ungated on purpose


def test_no_differential_when_the_rival_is_normal():
    agent = make_agent("Lung Cancer", status="partial", tools={
        "image": tool("Normal", probs(0.9, 0.05, 0.05)),
        "text": tool("Lung Cancer", probs(0.1, 0.1, 0.8))})
    assert not any("versus" in t for t in texts(plan_queries(agent)))


def test_query_cap_and_dedupe(pneumonia_agent):
    plan = plan_queries(pneumonia_agent)
    assert len(plan["queries"]) <= MAX_QUERIES
    assert len(set(texts(plan))) == len(plan["queries"])


def test_planner_never_changes_its_input(pneumonia_agent):
    before = copy.deepcopy(pneumonia_agent)
    plan = plan_queries(pneumonia_agent)
    assert pneumonia_agent == before and plan["predicted_class"] == "Pneumonia"


def test_every_query_phrase_has_topic_terms():
    """A phrase without TERMS would silently skip the topic gate, so every phrase must be covered."""
    blood_phrases = {p for _, _, lo, hi in BLOOD_RANGES.values() for p in (lo, hi) if p}
    assert blood_phrases <= set(TERMS), blood_phrases - set(TERMS)
    assert set(IMAGE_PHRASES.values()) <= set(TERMS), set(IMAGE_PHRASES.values()) - set(TERMS)


def test_class_phrases_cover_exactly_the_disease_classes():
    assert set(planner.CLASS_PHRASE) == {"Pneumonia", "Lung Cancer"}
