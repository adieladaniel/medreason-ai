"""
Runs the tools, belief state, consistency checker, and planner for one case.

Design rule this preserves: the orchestrator calls tools and reasons about their
outputs, it never produces a class prediction of its own. The predicted class in the
result is always belief.pool().argmax(), which traces back to the saved Phase 2 models
through the exact rule validated in Phase 3.

CLI (mirrors explain/run.py, for testing against real validation cases and for building
the Phase 8b demo-case set):
    python -m app.agent.orchestrator --study-id <STUDY_ID>
    python -m app.agent.orchestrator --study-id <STUDY_ID> --drop blood
    python -m app.agent.orchestrator --study-id <STUDY_ID> --drop blood --drop image
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import argparse
import json
from pathlib import Path

import pandas as pd

from app.agent.reasoning import reason  # noqa: F401  (re-exported: the shared belief, consistency and plan step)
from app.agent.tools import MODALITY_PRIORITY, TOOLS
from explain.blood import RAW_LABS
from explain.text import extract_indication


def run_case(image_path: str | None = None, labs: dict | None = None,
             indication: str | None = None) -> dict:
    inputs = {"image": image_path, "blood": labs, "text": indication}
    provided = {m: v for m, v in inputs.items() if v is not None and v != {}}
    if not provided:
        raise ValueError("run_case needs at least one of image_path, labs, indication")

    tool_outputs = {}
    for modality in MODALITY_PRIORITY:
        if modality in provided:
            tool_outputs[modality] = TOOLS[modality](provided[modality])

    reasoned = reason({m: r["probabilities"] for m, r in tool_outputs.items()})
    return {
        "belief": reasoned["belief"],
        "tool_outputs": tool_outputs,
        "consistency": reasoned["consistency"],
        "missing_info_plan": reasoned["missing_info_plan"],
    }


def _case_from_split(study_id: int, split: str, drop: set[str]) -> dict:
    df = pd.read_csv(f"data/processed/{split}.csv")
    row = df.loc[df["study_id"] == study_id]
    if row.empty:
        raise SystemExit(f"study_id {study_id} not in {split}.csv")
    r = row.iloc[0]
    report = Path(r["report_path"]).read_text(encoding="utf-8", errors="ignore")
    return {
        "true_label": r["final_label"],
        "image_path": None if "image" in drop else r["image_path"],
        "labs": None if "blood" in drop else {c: (None if pd.isna(r[c]) else float(r[c])) for c in RAW_LABS},
        "indication": None if "text" in drop else extract_indication(report),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--drop", action="append", default=[], choices=["image", "blood", "text"],
                     help="simulate a missing modality; repeatable")
    args = ap.parse_args()

    case = _case_from_split(args.study_id, args.split, set(args.drop))
    print(f"study {args.study_id}  true label: {case['true_label']}"
          f"  (dropped: {sorted(args.drop) or 'none'})")

    out = run_case(image_path=case["image_path"], labs=case["labs"], indication=case["indication"])

    print("\n--- belief, step by step ---")
    for step in out["belief"]["history"]:
        print(f"  + {step['modality']:<6} -> {step['probabilities']}  "
              f"belief now: {step['belief_after']}  ({step['predicted_class']})")

    b = out["belief"]
    print(f"\nfinal prediction: {b['predicted_class']}  (confidence {b['confidence']:.0%})")
    print(f"probabilities: {b['probabilities']}")
    print(f"\nconsistency: {out['consistency']['status']}")
    print(f"  {out['consistency']['detail']}")
    print(f"\nmissing-information plan: {out['missing_info_plan']}")

    dump = {k: v for k, v in out.items() if k != "tool_outputs"}
    dump["tool_outputs"] = {m: {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
                             for m, r in out["tool_outputs"].items()}
    print("\n--- full result (JSON) ---")
    print(json.dumps(dump, indent=2))
