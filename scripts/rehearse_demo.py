"""Rehearse the demo: send the nine frozen example cases through a RUNNING stack with the real models and check each still shows
what its note says.

Why: the demo is nine cases chosen by rule (scripts/build_demo_cases.py), including ones the system gets wrong, and each carries a note
for the presenter ("the models disagree and the pooled answer is still right", "wrong, but flagged as uncertain"). A model file, a
setting, a library version or Ollama being down can change an answer or an explanation without any test failing, and a demo is the
worst place to find out. tests/test_demo_cases.py reproduces the frozen predictions in-process; this does what a presenter does: it
goes through the running service over HTTP, the same way the page's example menu and form do, with Ollama and the evidence index, and
looks at the whole result. scripts/smoke_stack.py asks whether a stack is alive; this asks whether the demo will work.

What each case must show (the manifest freezes it, demo/cases/manifest.json, git-ignored because it holds MIMIC text):
  the case ends "done", the predicted class, the consistency verdict, which inputs were used, the low-confidence flag, the prediction
  set and the cancer alert all equal the frozen values, and the pooled confidence is within 0.01 of it. The two built-in pictures that are
  not X-rays must be refused before any model runs. It prints the time each case took, so a slow first case (models still loading) is
  visible, and the sentences the evidence note verified or dropped. Those two vary between runs (the writer is not reproducible), so they
  are reported and never compared.

The readiness check is strict: Ollama must have its model and Qdrant its passages, or the evidence part of the demo would silently be
empty. It never touches the sealed test split and needs nothing beyond the standard library and the demo folder.

  python scripts/rehearse_demo.py [--url http://127.0.0.1:8000] [--allow-degraded] [--keep] [--wait 300]
Exit status 0 when every case matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_stack as smoke  # noqa: E402  (the HTTP helpers and the readiness check are shared, not copied)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "demo" / "cases" / "manifest.json"
CONFIDENCE_TOLERANCE = 0.01
BUILTIN = ("not-an-xray-photo", "not-an-xray-noise")


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    except (OSError, ValueError, KeyError) as err:
        raise smoke.Failed(f"cannot read the demo manifest ({path}): {err}. Build it with python scripts/build_demo_cases.py") from err
    if not cases:
        raise smoke.Failed("the demo manifest has no cases")
    return cases


def shown(view: dict) -> dict:
    """The fields of a stored case that the page headlines, in the shape the manifest freezes them."""
    pred = view.get("prediction") or {}
    belief, decision = pred.get("belief") or {}, pred.get("decision") or {}
    alert = decision.get("cancer_alert") or {}
    return {"predicted_class": belief.get("predicted_class"), "pooled_confidence": max((belief.get("probabilities") or {0: 0}).values()),
            "consistency": (pred.get("consistency") or {}).get("status"), "inputs": sorted(pred.get("modalities_used") or []),
            "low_confidence": bool(decision.get("low_confidence")), "prediction_set": decision.get("prediction_set") or [],
            "cancer_alert": bool(alert.get("informative") and alert.get("flag"))}


def compare(expected: dict, view: dict) -> list[str]:
    """What differs between the frozen values and a finished case, one sentence each. Empty means it shows what its note says."""
    problems = []
    if view.get("status") != "done":
        problems.append(f"ended {view.get('status')!r} at stage {view.get('stage')!r}: {view.get('error')}")
    got = shown(view)
    for key in ("predicted_class", "consistency", "inputs", "low_confidence", "cancer_alert"):
        if got[key] != expected[key]:
            problems.append(f"{key} is {got[key]!r}, frozen as {expected[key]!r}")
    if sorted(got["prediction_set"]) != sorted(expected["prediction_set"]):
        problems.append(f"prediction_set is {got['prediction_set']!r}, frozen as {expected['prediction_set']!r}")
    if abs(got["pooled_confidence"] - expected["pooled_confidence"]) > CONFIDENCE_TOLERANCE:
        problems.append(f"pooled_confidence is {got['pooled_confidence']:.3f}, frozen as {expected['pooled_confidence']:.3f}")
    return problems


def evidence_counts(view: dict) -> str:
    ev = view.get("evidence") or {}
    final = ev.get("final") or {}
    if not final:
        return "no evidence note"
    return f"{len(final.get('verified') or [])} verified, {len(final.get('dropped') or [])} dropped"


def submit_like_the_page(base: str, example_id: str) -> str:
    """Load the example the way the Analyze page's menu does, then submit it the way the form does. Returns the case id."""
    status, example = smoke.get_json(base, f"/api/examples/{example_id}")
    if status != 200:
        raise smoke.Failed(f"/api/examples/{example_id} answered {status}")
    img_status, image = smoke.call(base, "GET", f"/api/examples/{example_id}/image")
    if img_status != 200 or not image:
        raise smoke.Failed(f"/api/examples/{example_id}/image answered {img_status}")
    fields = {"indication": example.get("indication", "")}
    if example.get("labs"):
        fields["labs"] = json.dumps(example["labs"])
    body, ctype = smoke.form(fields, {"image": ("example.jpg", image, "image/jpeg")})
    status, raw = smoke.call(base, "POST", "/api/cases", body, ctype, timeout=60)
    if status != 202:
        raise smoke.Failed(f"POST /api/cases answered {status}: {raw[:200]!r}")
    return json.loads(raw)["id"]


def run_case(base: str, case: dict, wait: float, keep: bool) -> tuple[list[str], float, str]:
    started = time.monotonic()
    cid = submit_like_the_page(base, case["id"])
    view: dict = {}
    while time.monotonic() - started < wait:
        _, view = smoke.get_json(base, f"/api/cases/{cid}")
        if view.get("status") in smoke.TERMINAL:
            break
        time.sleep(1)
    else:
        raise smoke.Failed(f"still {view.get('status')!r} at stage {view.get('stage')!r} after {wait:g} s")
    seconds = time.monotonic() - started
    problems = compare(case["expected"], view)
    note = evidence_counts(view)
    if not keep:
        smoke.call(base, "DELETE", f"/api/cases/{cid}")
    return problems, seconds, note


def check_builtin_pictures_are_refused(base: str, expected_ids: tuple[str, ...] = BUILTIN) -> str:
    _, listing = smoke.get_json(base, "/api/examples")
    offered = {e["id"] for e in listing["examples"]}
    missing = [i for i in expected_ids if i not in offered]
    if missing:
        raise smoke.Failed(f"not offered by the example menu: {', '.join(missing)}")
    before = smoke.case_ids(base)
    for example_id in expected_ids:
        _, image = smoke.call(base, "GET", f"/api/examples/{example_id}/image")
        body, ctype = smoke.form({}, {"image": ("picture.png", image, "image/png")})
        status, raw = smoke.call(base, "POST", "/api/cases", body, ctype)
        if status != 422:
            raise smoke.Failed(f"{example_id} was answered {status}, a picture that is not an X-ray must be refused with 422")
    if smoke.case_ids(base) != before:
        raise smoke.Failed("a refused picture was stored as a case")
    return f"{len(expected_ids)} pictures refused before any model ran, nothing stored"


def run(base: str, manifest: Path = MANIFEST, allow_degraded: bool = False, wait: float = 300, keep: bool = False, out=print) -> bool:
    ok = True
    try:
        cases = load_manifest(manifest)
        out(f"ok    health: {smoke.wait_for_health(base, 60)}")
        out(f"ok    ready: {smoke.check_ready(base, strict=not allow_degraded)}")
        listing = smoke.get_json(base, "/api/examples")[1]
        offered = {e["id"] for e in listing["examples"] if e.get("kind") == "validation case"}
        absent = [c["id"] for c in cases if c["id"] not in offered]
        if absent:
            raise smoke.Failed(f"the server does not offer these demo cases (is demo/cases mounted?): {', '.join(absent)}")
        out(f"ok    menu: all {len(cases)} cases offered")
    except smoke.Failed as err:
        out(f"FAIL  {err}")
        return False
    for i, case in enumerate(cases):
        try:
            problems, seconds, note = run_case(base, case, wait, keep)
        except smoke.Failed as err:
            ok = False
            out(f"FAIL  {case['id']}: {err}")
            continue
        if problems:
            ok = False
            out(f"FAIL  {case['id']} ({seconds:.0f} s):")
            for p in problems:
                out(f"        {p}")
        else:
            out(f"ok    {case['id']:26} {seconds:5.1f} s  {case['expected']['predicted_class']:12} {note}" + ("   (first case: models and Ollama may still be loading)" if i == 0 and seconds > 30 else ""))
    try:
        out(f"ok    built-in pictures: {check_builtin_pictures_are_refused(base)}")
    except smoke.Failed as err:
        ok = False
        out(f"FAIL  built-in pictures: {err}")
    out("the demo will show what its notes say" if ok else "DO NOT DEMO until the lines marked FAIL are explained")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="address of the API (127.0.0.1, not localhost: see CLAUDE.md)")
    ap.add_argument("--allow-degraded", action="store_true", help="do not require Ollama and Qdrant (the evidence note will be empty)")
    ap.add_argument("--keep", action="store_true", help="leave the cases in the database (default: delete them afterwards)")
    ap.add_argument("--wait", type=float, default=300, help="seconds to wait for each case")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()
    sys.exit(0 if run(args.url.rstrip("/"), args.manifest, args.allow_degraded, args.wait, args.keep) else 1)


if __name__ == "__main__":
    main()
