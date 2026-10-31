"""
Send one real validation case (its X-ray, its own clinical indication and its own labs) to a running server, so it
can be looked at on the Analyze page.

Why this exists: typing 13 lab values by hand is slow, and the page's "Fill example values" button gives one
made-up pneumonia-like set. Mixing that with a different patient's X-ray makes the text and blood models outvote
the image every time, which shows the fusion working and teaches nothing about the cases. Here every input comes
from the same study, as the models were built for.

Only the validation and training splits are allowed. The test split stays sealed until the final evaluation
(experiments/test_evaluation/PLAN.md), so asking for it is an error, not a warning.

Run from the repo root, with the server and Ollama already running:
    python scripts/submit_case.py --study-id <STUDY_ID>                 # a pneumonia case, all three inputs
    python scripts/submit_case.py --study-id <STUDY_ID>                 # a normal case
    python scripts/submit_case.py --study-id <STUDY_ID>                 # a lung cancer case the models get wrong
    python scripts/submit_case.py --study-id <STUDY_ID> --drop image    # same case without the X-ray
It prints the address to open. The case is also listed under "Recent cases" on the page.
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_request(case: dict, drop: set[str]) -> tuple[dict, dict | None]:
    """The form fields and the file part for one case from app.agent.orchestrator._case_from_split. A dropped
    input is simply left out, which is also how a real user leaves a field empty."""
    data = {}
    if "text" not in drop and case.get("indication"):
        data["indication"] = case["indication"]
    labs = {k: v for k, v in (case.get("labs") or {}).items() if v is not None}
    if "blood" not in drop and labs:
        data["labs"] = json.dumps(labs)
    files = None
    if "image" not in drop and case.get("image_path"):
        files = {"image": ("film.jpg", Path(case["image_path"]).read_bytes(), "image/jpeg")}
    return data, files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-id", type=int, required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"], help="the test split is deliberately not offered")
    ap.add_argument("--drop", action="append", default=[], choices=["image", "blood", "text"], help="leave an input out; repeatable")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    from app.agent.orchestrator import _case_from_split          # imported here: it loads the model libraries
    case = _case_from_split(args.study_id, args.split, set())
    data, files = build_request(case, set(args.drop))
    if not data and not files:
        sys.exit("nothing left to send after --drop")
    print(f"study {args.study_id}, true label: {case['true_label']} (sending {', '.join(sorted(set(['indication', 'labs', 'image']) & (set(data) | set(files or {}))))})")
    try:
        r = httpx.post(f"{args.server}/api/cases", data=data, files=files, timeout=60)
    except httpx.ConnectError:
        sys.exit(f"could not reach {args.server}. Start it with: python -m uvicorn app.main:app --port 8000")
    if r.status_code != 202:
        sys.exit(f"the server refused it ({r.status_code}): {r.text[:300]}")
    cid = r.json()["id"]
    print(f"accepted. Open this address to watch it:\n  {args.server}/#/analyze/{cid}")


if __name__ == "__main__":
    main()
