"""Phase 7 storage layer, on a throwaway SQLite file."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.database import repository as repo
from app.database.models import Case, Event, LlmCall
from app.database.session import init_db, session_factory, session_scope


@pytest.fixture
def url(tmp_path):
    u = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    init_db(u)
    return u


def test_jsonable_handles_what_the_pipeline_produces():
    out = repo.jsonable({"terms": {"b", "a"}, "p": np.float32(0.5), "v": np.array([1, 2]), 3: (1, 2),
                         "nan": float("nan"), "n": np.int64(7)})
    assert out == {"terms": ["a", "b"], "p": 0.5, "v": [1, 2], "3": [1, 2], "nan": None, "n": 7}
    with pytest.raises(TypeError):
        repo.jsonable({"x": object()})


def test_a_case_needs_at_least_one_input(url):
    with session_scope(url) as s:
        with pytest.raises(ValueError):
            repo.create_case(s)


def test_create_and_read_back_a_case(url):
    with session_scope(url) as s:
        c = repo.create_case(s, indication="cough and fever", labs={"WBC": np.float64(12.5)})
        cid = c.id
    with session_scope(url) as s:
        c = repo.get_case(s, cid)
        assert c.status == "queued" and c.stage == "queued" and c.labs == {"WBC": 12.5}
        assert c.prediction is None and c.timings == {}
        assert len(cid) == 32


def test_stage_results_and_timings_accumulate(url):
    with session_scope(url) as s:
        cid = repo.create_case(s, indication="x").id
    with session_scope(url) as s:
        c = repo.get_case(s, cid)
        repo.set_stage(s, c, "predicting")
        repo.save_result(s, c, "prediction", {"belief": {"predicted_class": "Pneumonia"}})
        repo.record_timing(s, c, "predicting", 4.2)
    with session_scope(url) as s:                       # a later transaction must not lose the earlier timing
        c = repo.get_case(s, cid)
        repo.record_timing(s, c, "explaining", 61.0)
        repo.save_result(s, c, "evidence", {"passages": [{"terms": {"albumin"}}]})
    with session_scope(url) as s:
        c = repo.get_case(s, cid)
        assert c.stage == "predicting" and c.status == "running"
        assert c.prediction["belief"]["predicted_class"] == "Pneumonia"
        assert c.timings == {"predicting": 4.2, "explaining": 61.0}
        assert c.evidence["passages"][0]["terms"] == ["albumin"]


def test_unknown_result_field_is_rejected(url):
    with session_scope(url) as s:
        c = repo.create_case(s, indication="x")
        with pytest.raises(ValueError):
            repo.save_result(s, c, "diagnosis", {})


def test_events_come_back_in_order_and_after_a_cursor(url):
    with session_scope(url) as s:
        c = repo.create_case(s, indication="x")
        ids = [repo.add_event(s, c, st, status).id for st, status in
               [("predicting", "started"), ("predicting", "finished"), ("explaining", "started")]]
        cid = c.id
    with session_scope(url) as s:
        assert [e.stage for e in repo.events_after(s, cid)] == ["predicting", "predicting", "explaining"]
        assert [e.id for e in repo.events_after(s, cid, after_id=ids[1])] == [ids[2]]
        assert repo.events_after(s, "nope") == []


def test_llm_calls_are_stored_with_their_timing(url):
    with session_scope(url) as s:
        c = repo.create_case(s, indication="x")
        repo.add_llm_call(s, c, model="llama3.2:latest", purpose="background_sentence", prompt="P", response="R", seconds=3.14159)
        cid = c.id
    with session_scope(url) as s:
        call = repo.get_case(s, cid).llm_calls[0]
        assert (call.model, call.prompt, call.response, call.seconds) == ("llama3.2:latest", "P", "R", 3.142)


def test_restart_recovery_fails_unfinished_cases_only(url):
    with session_scope(url) as s:
        a = repo.create_case(s, indication="a")                   # queued
        b = repo.create_case(s, indication="b")
        repo.set_stage(s, b, "explaining")                        # running
        d = repo.create_case(s, indication="d")
        d.status = "done"
        ids = (a.id, b.id, d.id)
    with session_scope(url) as s:
        assert repo.interrupted_to_failed(s) == 2
    with session_scope(url) as s:
        got = {i: repo.get_case(s, i) for i in ids}
        assert [got[i].status for i in ids] == ["failed", "failed", "done"]
        assert got[ids[1]].error == "interrupted by a server restart"
        assert repo.events_after(s, ids[1])[-1].status == "failed"


def test_purge_removes_old_cases_children_and_files(url, tmp_path):
    img = tmp_path / "old.png"
    img.write_bytes(b"x")
    with session_scope(url) as s:
        old = repo.create_case(s, indication="old", image_path=str(img))
        repo.add_event(s, old, "predicting", "started")
        repo.add_llm_call(s, old, model="m", purpose="p", prompt="a", response="b", seconds=1)
        old.created_at = datetime.now(timezone.utc) - timedelta(days=40)
        fresh = repo.create_case(s, indication="fresh")
        old_id, fresh_id = old.id, fresh.id
    with session_scope(url) as s:
        assert repo.purge_older_than(s, days=30) == 1
    with session_scope(url) as s:
        assert repo.get_case(s, old_id) is None and repo.get_case(s, fresh_id) is not None
        assert s.scalars(select(Event)).first() is None and s.scalars(select(LlmCall)).first() is None
    assert not img.exists()


def test_sqlite_enforces_foreign_keys(url):
    with pytest.raises(IntegrityError):
        with session_scope(url) as s:
            s.add(Event(case_id="does-not-exist", stage="predicting", status="started"))


def test_list_cases_is_newest_first_and_paged(url):
    with session_scope(url) as s:
        for i in range(5):
            c = repo.create_case(s, indication=f"case {i}")
            c.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    with session_scope(url) as s:
        assert [c.indication for c in repo.list_cases(s, limit=2)] == ["case 4", "case 3"]
        assert [c.indication for c in repo.list_cases(s, limit=2, offset=2)] == ["case 2", "case 1"]


def test_sessions_are_independent_per_url(tmp_path):
    a, b = (f"sqlite:///{(tmp_path / n).as_posix()}" for n in ("a.db", "b.db"))
    init_db(a)
    init_db(b)
    with session_scope(a) as s:
        repo.create_case(s, indication="only in a")
    with session_scope(b) as s:
        assert repo.list_cases(s) == []
    assert session_factory(a) is session_factory(a)
