"""
Unified explainability entrypoint -- run all three per-modality explainers for one case
and print/save a combined view. This is what the Phase 5 agent layer will call after the
models produce a prediction.

CLI:
  python -m explain.run --study-id 50262066            # pull image/labs/indication from val.csv
  python -m explain.run --study-id 50262066 --split val --save experiments/explain_samples/
  python -m explain.run --study-id 50262066 --only blood,text
"""
from __future__ import annotations

import torch  # noqa: F401  -- import before sklearn/shap on Windows (see CLAUDE.md §4)

import argparse
import json
from pathlib import Path

import pandas as pd

from explain import blood as blood_mod
from explain import text as text_mod
from explain import image as image_mod


def _case_from_split(study_id: int, split: str) -> dict:
    df = pd.read_csv(f"data/processed/{split}.csv")
    row = df.loc[df["study_id"] == study_id]
    if row.empty:
        raise SystemExit(f"study_id {study_id} not in {split}.csv")
    r = row.iloc[0]
    report = Path(r["report_path"]).read_text(encoding="utf-8", errors="ignore")
    return {
        "study_id": int(study_id),
        "true_label": r["final_label"],
        "image_path": r["image_path"],
        "indication": text_mod.extract_indication(report),
        "labs": {c: (None if pd.isna(r[c]) else float(r[c])) for c in blood_mod.RAW_LABS},
    }


def explain_case(case: dict, only: set[str] | None = None, save_dir: Path | None = None) -> dict:
    only = only or {"blood", "text", "image"}
    out: dict = {"study_id": case.get("study_id"), "true_label": case.get("true_label"),
                 "modalities": {}}

    sid = case["study_id"]
    if "blood" in only:
        r = blood_mod.explain_blood(case["labs"])
        if save_dir:
            r["plot"] = str(blood_mod.plot_blood(r, save_dir / f"blood_{sid}.png"))
        out["modalities"]["blood"] = r
    if "text" in only:
        r = text_mod.explain_text(case["indication"])
        if save_dir:
            r["plot"] = str(text_mod.plot_text(r, save_dir / f"text_{sid}.png"))
        r.pop("_raw", None)
        out["modalities"]["text"] = r
    if "image" in only:
        r = image_mod.explain_image(case["image_path"])
        if save_dir:
            r["plot"] = str(image_mod.plot_image(r, save_dir / f"image_{sid}.png"))
        r.pop("_cam", None); r.pop("_display", None)
        out["modalities"]["image"] = r
    return out


def _print(out: dict) -> None:
    print(f"\nstudy {out['study_id']}   true label: {out['true_label']}")
    for m, r in out["modalities"].items():
        print(f"\n=== {m.upper()} — predicts {r['predicted_class']}  {r['probabilities']}")
        print(f"    {r['summary']}")
        for it in r["attributions"]["items"][:6]:
            v = "" if it.get("value") in (None, "") else f" ({it['value']})"
            print(f"      {it['direction']:>8}  {it['feature']}{v}   effect {it['effect']:+.3f}")
        if r.get("plot"):
            print(f"    plot: {r['plot']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", default="blood,text,image")
    ap.add_argument("--save", default=None, help="directory to write per-modality PNGs + JSON")
    args = ap.parse_args()

    case = _case_from_split(args.study_id, args.split)
    print(f"indication: {case['indication']!r}")
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    out = explain_case(case, only=set(args.only.split(",")), save_dir=save_dir)
    _print(out)

    if save_dir:
        (save_dir / f"explain_{args.study_id}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nsaved JSON -> {save_dir / f'explain_{args.study_id}.json'}")
