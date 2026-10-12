"""The claim-checker comparison: chunking logic and the metric summary, with no model downloads."""
import numpy as np
import pytest

from app.rag import eval_checker, minicheck


def test_flan_chunks_split_on_word_budget_and_keep_whole_sentences():
    sents = ["word " * 200 + "end." for _ in range(5)]            # 201 words each, budget is 500
    chunks = minicheck._chunks("flan-t5-large", None, sents)
    assert len(chunks) == 3                                        # 2 + 2 + 1 sentences
    assert all(c.count("end.") in (1, 2) for c in chunks)          # never cuts inside a sentence
    assert " ".join(chunks).count("end.") == 5                     # nothing lost


def test_short_passage_is_one_chunk():
    assert len(minicheck._chunks("flan-t5-large", None, ["One short sentence.", "Another one."])) == 1


def test_sentence_splitter_handles_empty_input():
    assert minicheck._sentences("") == [""]
    assert minicheck._sentences("A. B c. D e.") == ["A.", "B c.", "D e."]


def test_real_labels_cover_all_items_and_mark_exactly_the_audited_negatives():
    assert set(eval_checker.REAL_LABELS) == set(range(41))
    negatives = {i for i, v in eval_checker.REAL_LABELS.items() if v == 0}
    borderline = {i for i, v in eval_checker.REAL_LABELS.items() if v is None}
    assert negatives == {27, 34, 35} and borderline == {17, 25}


def test_summarise_computes_rates_and_auroc_from_cached_scores():
    pairs = [{"id": f"hand:{i}", "set": "hand", "label": l, "claim": "", "passage": ""} for i, l in enumerate([1, 1, 1, 0, 0])]
    pairs += [{"id": f"real:{i}", "set": "real", "label": l, "claim": "", "passage": ""} for i, l in enumerate([1, 1, 0])]
    scores = [0.9, 0.8, 0.2, 0.1, 0.6, 0.9, 0.4, 0.49]              # hand: one faithful low; one unfaithful high
    cache = {"x": {p["id"]: {"score": s, "seconds": 1.0} for p, s in zip(pairs, scores)}}
    r = eval_checker.summarise(["x"], pairs, cache)["x"]
    assert r["threshold"] == 0.5
    assert r["hand"]["false_reject_n"] == 1 and r["hand"]["caught_n"] == 1
    assert r["real"]["false_reject_n"] == 1 and r["real"]["caught_n"] == 1
    assert r["hand"]["auroc"] == pytest.approx(5 / 6)
    assert r["all"]["n"] == 8


def test_current_checker_keeps_its_tuned_threshold():
    assert eval_checker.THRESHOLD["deberta-nli"] == 0.30
