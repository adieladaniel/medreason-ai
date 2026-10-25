"""The lookup cache in app/rag/retrieve.py. The retriever is replaced by a counting fake, so the reranker
(nearly all of a lookup's time) never runs."""
import pytest

import app.rag.retrieve as R


def result(cid, score):
    return {"chunk_id": cid, "text": f"passage {cid}", "rerank_score": score}


class FakeRetriever:
    def __init__(self, table):
        self.table, self.calls = table, []

    def search(self, query, k, mode, rerank, rerank_depth):
        self.calls.append(query)
        return [dict(r) for r in self.table.get(query, [])]


@pytest.fixture
def fake(monkeypatch):
    f = FakeRetriever({"pneumonia with low albumin": [result(1, 12.0), result(2, 4.0), result(3, 3.9)]})
    monkeypatch.setattr(R, "retriever", lambda: f)
    R.clear_cache()
    yield f
    R.clear_cache()


def test_a_repeated_query_is_answered_without_searching_again(fake):
    first = R.retrieve("pneumonia with low albumin")
    second = R.retrieve("pneumonia with low albumin")
    assert first == second and fake.calls == ["pneumonia with low albumin"]
    assert R.cache_info() == {"hits": 1, "misses": 1, "size": 1, "max_size": R.CACHE_SIZE}


def test_the_threshold_is_applied_before_the_answer_is_kept(fake):
    """A passage the threshold removed must never come back from the cache."""
    for _ in range(2):
        assert [r["chunk_id"] for r in R.retrieve("pneumonia with low albumin")] == [1, 2]      # 3.9 is under 4.0


def test_no_passage_is_a_cached_answer_too(fake):
    assert R.retrieve("something the library lacks") == [] and R.retrieve("something the library lacks") == []
    assert fake.calls == ["something the library lacks"]


def test_changing_a_returned_passage_does_not_change_the_next_answer(fake):
    R.retrieve("pneumonia with low albumin")[0]["text"] = "tampered"
    again = R.retrieve("pneumonia with low albumin")
    assert again[0]["text"] == "passage 1"


def test_different_k_is_a_different_entry(fake):
    assert len(R.retrieve("pneumonia with low albumin", k=1)) == 1
    assert len(R.retrieve("pneumonia with low albumin", k=5)) == 2
    assert len(fake.calls) == 2


def test_the_least_recently_used_answer_is_dropped_first(fake, monkeypatch):
    monkeypatch.setattr(R, "CACHE_SIZE", 2)
    R.retrieve("a")
    R.retrieve("b")
    R.retrieve("a")                     # a is now the most recent
    R.retrieve("c")                     # evicts b
    fake.calls.clear()
    R.retrieve("a")
    R.retrieve("b")
    assert fake.calls == ["b"]


def test_clearing_forgets_everything(fake):
    R.retrieve("pneumonia with low albumin")
    R.clear_cache()
    R.retrieve("pneumonia with low albumin")
    assert len(fake.calls) == 2 and R.cache_info()["misses"] == 1
