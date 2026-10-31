"""The progress dashboard's data file must not depend on any document or on git-ignored data: phase status is a small tracked file, and a rebuild
on a machine without the data (a fresh clone) keeps the sections the repository ships instead of replacing them with nothing."""
import json
from pathlib import Path

from app import dashboard_data as d

ROOT = Path(__file__).resolve().parent.parent


def test_phase_status_comes_from_a_tracked_data_file_not_from_a_document():
    phases = d._phases()
    assert [p["num"] for p in phases] == list(range(9))
    assert all(p["state"] in {"done", "partial", "todo"} and p["title"] and p["blurb"] for p in phases)
    assert "ROADMAP" not in (ROOT / "app" / "dashboard_data.py").read_text(encoding="utf-8")
    assert "ROADMAP" not in (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "app/data/phases.json" in d.WATCHED and "ROADMAP.md" not in d.WATCHED


def test_the_remaining_work_table_lists_what_is_actually_not_done():
    text = json.dumps(d.REMAINING).lower()
    assert len(d.REMAINING) >= 3 and all({"phase", "work", "effort"} <= set(r) for r in d.REMAINING)
    assert "backend" not in text and "sealed test-set evaluation" not in text          # these were done long ago


def test_a_section_missing_from_a_rebuild_keeps_the_shipped_one(monkeypatch, tmp_path):
    shipped = {"phases": [1], "data": {"splits": 3}, "models": {"a": 1}, "rag": None}
    out = tmp_path / "dashboard.json"
    out.write_text(json.dumps(shipped), encoding="utf-8")
    monkeypatch.setattr(d, "OUT_PATH", out)
    rebuilt = {"phases": [1, 2], "data": None, "models": {"a": 2}, "rag": None, "extra": None}
    kept = d.keep_existing(rebuilt)
    assert kept["data"] == {"splits": 3}                      # no data/ on this machine: keep what the repository ships
    assert kept["phases"] == [1, 2] and kept["models"] == {"a": 2}   # sections that were rebuilt are used
    assert kept["rag"] is None and kept["extra"] is None      # nothing shipped, nothing to keep


def test_without_a_shipped_file_the_rebuild_is_used_as_it_is(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "OUT_PATH", tmp_path / "missing.json")
    payload = {"data": None, "models": {"a": 1}}
    assert d.keep_existing(payload) == payload


def test_an_unreadable_shipped_file_does_not_stop_a_rebuild(monkeypatch, tmp_path):
    out = tmp_path / "dashboard.json"
    out.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(d, "OUT_PATH", out)
    assert d.keep_existing({"data": None}) == {"data": None}
