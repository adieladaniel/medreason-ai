"""
Runs one stored case through the stages and keeps the database current while it does.

What is saved, and when
  - before a stage: the case's `stage` is set and a "started" event is written
  - after a stage: its result goes into its own column, the stage's seconds go into `timings`, and a
    "finished" event is written. A client reading the case at any moment sees every result that exists.
  - a stage that works in pieces (explaining does one modality at a time) saves each piece through
    `progress`, so blood and image explanations are readable while the text SHAP run is going.
  - after the last stage: the case is "done". If a stage after the first fails, the case is "partial":
    the prediction and everything before the failure stay, `error` says what failed, and the later
    stages are skipped. If the first stage fails there is no answer, so the case is "failed".

No database session is open while a model runs. Each step opens a short session, writes and closes, so
a long explainer run never holds a connection, and SQLite's one writer is never blocked by it.

JobRunner is the in-process worker (one thread, one job at a time) that the API will submit case ids to.
One at a time is deliberate: there is one machine and the models do not fit in memory twice. The
database holds the state, so this could be swapped for a real queue later without changing the API. A job
does not survive a restart; repository.interrupted_to_failed marks such cases failed at startup.

CLI, to run one real validation case through the whole thing and print what was stored:
    python -m app.pipelines.runner --study-id 53383543 [--drop image|blood|text] [--db sqlite:///x.db]
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch before sklearn/shap, see CLAUDE.md section 4)

import logging
import queue
import threading
import time
from pathlib import Path

from app.config import get_settings
from app.database import repository as repo
from app.database.session import init_db, session_scope
from app.pipelines.staged import default_plotters, default_stages, provided_inputs

log = logging.getLogger("medreason.runner")


def _make_recorder(case_id: str, url: str | None):
    def record(**call) -> None:
        with session_scope(url) as s:
            repo.add_llm_call(s, repo.get_case(s, case_id), **call)
    return record


def _make_progress(case_id: str, field: str, url: str | None):
    def progress(data) -> None:
        with session_scope(url) as s:
            repo.save_result(s, repo.get_case(s, case_id), field, data)
    return progress


def run_job(case_id: str, url: str | None = None, stages=None, deps: dict | None = None,
            figure_dir: Path | None = None) -> str:
    """Runs every stage for the stored case and returns its final status."""
    stages = stages if stages is not None else default_stages()
    deps = dict(deps or {})
    figure_root = Path(figure_dir) if figure_dir is not None else get_settings().upload_dir

    with session_scope(url) as s:
        case = repo.get_case(s, case_id)
        if case is None:
            raise KeyError(f"no case {case_id}")
        inputs = provided_inputs(case.indication, case.labs, case.image_path)
        if not inputs:
            repo.fail_case(s, case, "the case has no usable input (no image, no lab values, no indication)")
            repo.add_event(s, case, "queued", "failed", detail={"error": "no usable input"})
            return "failed"
        repo.set_stage(s, case, stages[0].name)

    ctx = {"case_id": case_id, "inputs": inputs, "deps": deps, "figure_dir": figure_root / case_id,
           "recorder": _make_recorder(case_id, url)}
    if "plotters" not in deps and not deps.get("no_figures"):
        try:
            deps["plotters"] = default_plotters()
        except Exception:                                      # noqa: BLE001  (figures are optional)
            log.exception("could not load the figure functions; continuing without figures")

    for i, stage in enumerate(stages):
        with session_scope(url) as s:
            case = repo.get_case(s, case_id)
            repo.set_stage(s, case, stage.name)
            repo.add_event(s, case, stage.name, "started")
        ctx["progress"] = _make_progress(case_id, stage.field, url)
        t0 = time.perf_counter()
        try:
            data = stage.fn(ctx)
        except Exception as exc:                               # noqa: BLE001  (any stage failure is recorded, not raised)
            seconds = time.perf_counter() - t0
            message = f"{type(exc).__name__}: {exc}"
            log.exception("case %s: stage %s failed", case_id, stage.name)
            with session_scope(url) as s:
                case = repo.get_case(s, case_id)
                repo.record_timing(s, case, stage.name, seconds)
                repo.add_event(s, case, stage.name, "failed", seconds, {"error": message})
                if i == 0:
                    repo.fail_case(s, case, f"{stage.name} failed: {message}")
                    return "failed"
                repo.finish_case(s, case, error=f"{stage.name} failed: {message}")
                return "partial"
        seconds = time.perf_counter() - t0
        ctx[stage.field] = data
        with session_scope(url) as s:
            case = repo.get_case(s, case_id)
            repo.save_result(s, case, stage.field, data)
            repo.record_timing(s, case, stage.name, seconds)
            repo.add_event(s, case, stage.name, "finished", seconds)

    with session_scope(url) as s:
        repo.finish_case(s, repo.get_case(s, case_id))
    return "done"


_STOP = object()


class JobRunner:
    """One worker thread that runs submitted case ids in order."""

    def __init__(self, url: str | None = None, stages=None, deps: dict | None = None, warm_up: bool = False,
                 figure_dir: Path | None = None):
        # figure_dir is where each case's explanation figures go (one sub-folder per case). The API serves
        # them from the same place, so the app passes its upload folder to both.
        self.url, self.stages, self.deps, self._warm_up, self.figure_dir = url, stages, deps, warm_up, figure_dir
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self.warm_timings: dict | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="case-worker", daemon=True)
        self._thread.start()

    def submit(self, case_id: str) -> None:
        self._queue.put(case_id)

    def pending(self) -> int:
        return self._queue.qsize()

    def stop(self, timeout: float | None = None) -> None:
        """Lets the job in progress finish, then ends the worker."""
        self._queue.put(_STOP)
        if self._thread:
            self._thread.join(timeout)

    def _loop(self) -> None:
        if self._warm_up:                                      # in the worker, so the API starts at once
            from app.pipelines.staged import warm_up
            try:
                self.warm_timings = warm_up()
                log.info("models loaded: %s", self.warm_timings)
            except Exception:                                  # noqa: BLE001
                log.exception("warm-up failed; models will load on first use")
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                run_job(item, self.url, self.stages, self.deps, self.figure_dir)
            except Exception:                                  # noqa: BLE001  (run_job records its own failures; this is the last guard)
                log.exception("case %s: the worker could not run the job", item)


# ------------------------------------------------------------------ CLI
def _main() -> None:
    import argparse
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--drop", action="append", default=[], choices=["image", "blood", "text"])
    ap.add_argument("--db", default=None, help="database URL; default is the configured one")
    args = ap.parse_args()

    from app.agent.orchestrator import _case_from_split
    case = _case_from_split(args.study_id, args.split, set(args.drop))
    url = args.db or get_settings().database_url
    init_db(url)
    with session_scope(url) as s:
        c = repo.create_case(s, indication=case["indication"], labs=case["labs"], image_path=case["image_path"])
        case_id = c.id
    print(f"study {args.study_id}  true label: {case['true_label']}  (dropped: {sorted(args.drop) or 'none'})  case {case_id}")

    t0 = time.perf_counter()
    status = run_job(case_id, url)
    print(f"\nstatus: {status}   total {time.perf_counter() - t0:.1f} s")

    with session_scope(url) as s:
        c = repo.get_case(s, case_id)
        print("timings (s):", c.timings)
        for e in repo.events_after(s, case_id):
            print(f"  {e.ts:%H:%M:%S}  {e.stage:<11} {e.status:<9} {'' if e.seconds is None else f'{e.seconds:.1f} s'}")
        if c.error:
            print("error:", c.error)
        if c.prediction:
            b = c.prediction["belief"]
            print(f"\nprediction: {b['predicted_class']} ({b['confidence']:.0%}), used {c.prediction['modalities_used']}, "
                  f"per modality seconds {c.prediction['seconds']}")
            print("consistency:", c.prediction["consistency"]["status"], "|", c.prediction["consistency"]["detail"])
        if c.explanations:
            print("explanations match the stored prediction:",
                  {m: e["matches_prediction"] for m, e in c.explanations.items()},
                  "| figures:", {m: e.get("figure") for m, e in c.explanations.items()})
        if c.evidence and c.evidence.get("final"):
            print("\nwritten explanation:")
            for line in c.evidence["final"]["text"]:
                print(" -", line)
        print(f"\nLLM calls stored: {len(c.llm_calls)}")


if __name__ == "__main__":
    _main()
