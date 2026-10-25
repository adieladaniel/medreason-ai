"""The percentile helper behind the retrieval timing. The timing run itself needs the models and is not a test."""
import pytest

from app.rag import eval_latency


def test_summary_of_known_values():
    s = eval_latency.summarise([1.0, 2.0, 3.0, 4.0, 10.0])
    assert s == {"n": 5, "mean": 4.0, "p50": 3.0, "p95": 8.8, "max": 10.0, "min": 1.0}


def test_p95_sits_between_the_two_slowest_runs():
    s = eval_latency.summarise(list(range(1, 21)))          # 1..20 seconds
    assert s["p50"] == pytest.approx(10.5) and 19 < s["p95"] < 20


def test_empty_sample_does_not_crash():
    assert eval_latency.summarise([]) == {"n": 0}


def test_timed_measures_the_call_and_passes_arguments():
    seen = []
    t = eval_latency.timed(lambda a, b=0: seen.append((a, b)), 1, b=2)
    assert seen == [(1, 2)] and t >= 0
