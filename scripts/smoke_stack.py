"""Smoke test of a running MedReason-AI stack: is it up, is it wired to its database, and does a case go in and come out.

Why: the unit tests run the code, and the image build runs the unit tests, but neither starts the containers. A wrong environment
variable, a missing file in the image, a migration that fails on Postgres or a port that is not published all pass both and only show
when the stack runs. This script asks a running stack the questions a person would ask in the first minute. It needs nothing but the
standard library, so it runs on a bare CI runner as well as on this laptop.

What it checks, over HTTP only (the CI job checks the database side with psql):
  health      the process answers
  ready       the database and the worker are fine; Qdrant or Ollama missing is "degraded" and allowed unless --strict
  pages       the page, its scripts and its typeface are served from the image
  input-spec  the form description the Analyze page is built from
  bad input   an unknown lab name and an empty submission are refused with 422, and nothing is stored
  case        a case is accepted, finishes in a terminal state with an event trail, reveals no server path or traceback, and can be
              deleted (204, then 404)

What it does not check: that the answer is right. A CI runner has no trained models (they are derived from MIMIC and kept out of git), so
there a case ends "failed" at its first stage, and that is the expected, checked outcome (--expect failed). On a laptop with the real
models use --strict --expect done. Accuracy of the real models on real cases is scripts/run_e2e.py.

  python scripts/smoke_stack.py [--url http://127.0.0.1:8000] [--expect any|done|partial|failed] [--strict] [--wait 120]
Exit status 0 when every check passed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

TERMINAL = ("done", "partial", "failed")
LEAKS = ("/app/", "Traceback", "site-packages", "\\Users\\")       # nothing a visitor to an open API should read


class Failed(Exception):
    pass


def call(base: str, method: str, path: str, body: bytes | None = None, content_type: str | None = None, timeout: float = 15):
    """(status, body bytes). HTTP error statuses are returned, not raised; an unreachable server raises Failed."""
    req = urllib.request.Request(base + path, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        raise Failed(f"{method} {path}: {getattr(err, 'reason', err)}") from err


def form(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]] | None = None) -> tuple[bytes, str]:
    """A multipart/form-data body: text fields, and optionally files as {field: (file name, bytes, content type)}."""
    boundary = uuid.uuid4().hex
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    for name, (filename, data, ctype) in (files or {}).items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def get_json(base: str, path: str) -> tuple[int, dict]:
    status, body = call(base, "GET", path)
    try:
        return status, json.loads(body)
    except ValueError:
        raise Failed(f"GET {path} answered {status} with something that is not JSON") from None


# ------------------------------------------------------------------ the checks. Each returns a short detail or raises Failed.
def wait_for_health(base: str, seconds: float) -> str:
    deadline, last = time.monotonic() + seconds, "no answer"
    while time.monotonic() < deadline:
        try:
            status, _ = call(base, "GET", "/api/health", timeout=5)
            if status == 200:
                return f"answered after {seconds - (deadline - time.monotonic()):.0f} s"
            last = f"status {status}"
        except Failed as err:
            last = str(err)
        time.sleep(2)
    raise Failed(f"/api/health did not answer within {seconds:g} s (last: {last})")


def check_ready(base: str, strict: bool) -> str:
    status, body = get_json(base, "/api/ready")
    checks = body.get("checks", {})
    for name in ("database", "worker"):
        if not checks.get(name, {}).get("ok"):
            raise Failed(f"{name} is not ok: {checks.get(name, {}).get('detail', 'missing from the answer')} (overall {body.get('status')})")
    overall = body.get("status")
    if overall not in ("ok", "degraded") or status != 200:
        raise Failed(f"overall status {overall!r} with HTTP {status}")
    missing = sorted(n for n, c in checks.items() if not c.get("ok"))
    if strict and missing:
        raise Failed(f"--strict, and not ok: {', '.join(missing)}")
    return overall + (f" (missing: {', '.join(missing)})" if missing else "")


def check_pages(base: str) -> str:
    status, page = call(base, "GET", "/")
    if status != 200 or b"MedReason-AI" not in page:
        raise Failed(f"/ answered {status} without the page")
    assets = ["/static/app.js", "/static/analyze.js", "/static/shell.js", "/static/style.css", "/static/vendor/inter/inter-latin-wght-normal.woff2"]
    bad = [a for a in assets if call(base, "GET", a)[0] != 200]
    if bad:
        raise Failed(f"not served: {', '.join(bad)}")
    return f"page and {len(assets)} files"


def check_input_spec(base: str) -> str:
    status, spec = get_json(base, "/api/input-spec")
    labs = spec.get("labs", [])
    if status != 200 or len(labs) != 17 or not all(lab.get("name") and lab.get("unit") for lab in labs):
        raise Failed(f"status {status}, {len(labs)} labs")
    return "17 labs with units"


def case_ids(base: str) -> list[str]:
    status, body = get_json(base, "/api/cases?limit=100")             # newest first, so a stored submission would change this list
    if status != 200:
        raise Failed(f"GET /api/cases answered {status}")
    return [c["id"] for c in body["cases"]]


def check_bad_input(base: str) -> str:
    before = case_ids(base)
    body, ctype = form({"labs": json.dumps({"not_a_lab": 1})})
    first = call(base, "POST", "/api/cases", body, ctype)[0]
    body, ctype = form({})
    second = call(base, "POST", "/api/cases", body, ctype)[0]
    if (first, second) != (422, 422):
        raise Failed(f"an unknown lab gave {first} and an empty submission {second}, both should be 422")
    if case_ids(base) != before:
        raise Failed("a refused submission was stored anyway")
    return "both refused with 422, nothing stored"


def check_case(base: str, expect: str, wait: float) -> str:
    body, ctype = form({"indication": "Fever and cough for three days, evaluate for pneumonia."})
    status, raw = call(base, "POST", "/api/cases", body, ctype)
    if status != 202:
        raise Failed(f"POST /api/cases answered {status}: {raw[:200]!r}")
    cid = json.loads(raw)["id"]
    deadline, view = time.monotonic() + wait, {}
    while time.monotonic() < deadline:
        _, view = get_json(base, f"/api/cases/{cid}")
        if view.get("status") in TERMINAL:
            break
        time.sleep(1)
    else:
        raise Failed(f"case {cid} was still {view.get('status')!r} at stage {view.get('stage')!r} after {wait:g} s")
    final = view["status"]
    if expect != "any" and final != expect:
        raise Failed(f"case ended {final!r}, expected {expect!r} (error: {view.get('error')})")
    _, events = get_json(base, f"/api/cases/{cid}/events")
    if not events.get("events"):
        raise Failed("a finished case has no events")
    everything = json.dumps(view) + json.dumps(events)
    leaked = [x for x in LEAKS if x in everything]
    if leaked:
        raise Failed(f"the case view reveals {leaked}")
    if call(base, "DELETE", f"/api/cases/{cid}")[0] != 204:
        raise Failed("DELETE did not answer 204")
    if call(base, "GET", f"/api/cases/{cid}")[0] != 404:
        raise Failed("a deleted case can still be read")
    return f"ended {final} with {len(events['events'])} events, nothing leaked, deleted"


def run(base: str, expect: str = "any", strict: bool = False, wait: float = 120, out=print) -> bool:
    """Run every check, print one line each, return True when all passed. A check that needs the server stops the run if it is down."""
    ok = True

    def step(name: str, fn, *args) -> bool:
        nonlocal ok
        try:
            out(f"ok    {name}: {fn(*args)}")
            return True
        except Failed as err:
            ok = False
            out(f"FAIL  {name}: {err}")
            return False

    if not step("health", wait_for_health, base, wait):
        return False
    step("ready", check_ready, base, strict)
    step("pages", check_pages, base)
    step("input-spec", check_input_spec, base)
    step("bad input", check_bad_input, base)
    step("case", check_case, base, expect, wait)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="address of the API (use 127.0.0.1, not localhost: see CLAUDE.md)")
    ap.add_argument("--expect", default="any", choices=["any", *TERMINAL], help="the state a case must end in (CI has no models: failed)")
    ap.add_argument("--strict", action="store_true", help="fail unless every dependency (Qdrant, Ollama) is ok too")
    ap.add_argument("--wait", type=float, default=120, help="seconds to wait for the API to come up and for the case to finish")
    args = ap.parse_args()
    sys.exit(0 if run(args.url.rstrip("/"), args.expect, args.strict, args.wait) else 1)


if __name__ == "__main__":
    main()
