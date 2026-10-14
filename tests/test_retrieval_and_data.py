"""Pure retrieval helpers, and guards on the committed dashboard data."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_rrf_rewards_agreement_between_rankings():
    from app.rag.hybrid import rrf
    fused = rrf([[1, 2, 3], [2, 1, 3, 4]])
    assert fused[1] == pytest.approx(fused[2], rel=0.02)     # swapped ranks 1 and 2: near tie
    assert fused[3] > fused[4]                               # in both lists beats in one list
    assert set(fused) == {1, 2, 3, 4}


def test_bm25_prefers_the_passage_with_the_query_words():
    from app.rag.hybrid import BM25, tokenize
    docs = [tokenize(t) for t in ["elevated LDH predicts survival in lung cancer",
                                  "chest radiograph findings in community acquired pneumonia",
                                  "albumin and nutrition in hospitalised patients"]]
    scores = BM25(docs).scores(tokenize("lactate LDH survival"))
    assert int(scores.argmax()) == 0


def test_tokenizer_stems_so_inflections_match():
    from app.rag.hybrid import tokenize
    assert tokenize("infections") == tokenize("infection")
    assert tokenize("elevated") == tokenize("elevate")


def test_committed_dashboard_data_holds_aggregates_only():
    """CLAUDE.md rule: the committed JSON has no study, subject or admission identifiers."""
    p = ROOT / "app" / "data" / "dashboard.json"
    if not p.exists():
        pytest.skip("dashboard.json not built")
    raw = p.read_text(encoding="utf-8")
    for key in ("study_id", "subject_id", "hadm_id", "image_path", "report_path"):
        assert key not in raw, key


def test_dashboard_builder_survives_missing_results(monkeypatch, tmp_path):
    from app import dashboard_data as d
    monkeypatch.setattr(d, "ROOT", tmp_path)
    assert d._safe(d._rag) is None


def test_corpus_manifest_only_lists_unique_papers_with_known_topics():
    m = json.loads((ROOT / "app" / "rag" / "corpus_sources.json").read_text(encoding="utf-8"))["sources"]
    ids = [s["pmcid"] for s in m]
    assert len(ids) == len(set(ids))
    assert {s["topic"] for s in m} <= {"pneumonia", "lung_cancer", "lung_cancer_labs", "differential"}


def test_gap_probe_counts_only_on_topic_hits_and_uses_the_fallback():
    from app.rag import eval_gaps
    calls = []

    def fake(query, k=2):
        calls.append(query)
        if query == "pneumonia chest X-ray with pulmonary edema":
            return []
        if query == "pulmonary edema versus pneumonia chest radiograph":
            return [{"text": "Pulmonary edema can mimic pneumonia.", "rerank_score": 5.1}]
        return [{"text": "Unrelated passage about something else.", "rerank_score": 9.0}]

    qs = [q for q in eval_gaps.planner_queries() if q["finding"] == "pulmonary edema" and q["text"].startswith("pneumonia")]
    rows = eval_gaps.probe(qs, fake)
    assert rows[0]["on_topic_hits"] == 1 and "pulmonary edema versus pneumonia chest radiograph" in calls
    lab = eval_gaps.probe([{"finding": "low albumin", "text": "pneumonia with low albumin", "fallback": None}], fake)
    assert lab[0]["hits"] == 1 and lab[0]["on_topic_hits"] == 0      # retrieved, but not about albumin: still a gap


def test_planner_query_list_has_no_duplicates_and_covers_both_classes():
    from app.rag import eval_gaps
    qs = eval_gaps.planner_queries()
    texts = [q["text"] for q in qs]
    assert len(texts) == len(set(texts)) and any(t.startswith("lung cancer") for t in texts) and any(t.startswith("pneumonia") for t in texts)
