"""
Choose and freeze the example cases for the demo. Validation studies only; the sealed test split is never read here.

Why a script with rules and not hand-picking: a demo made of the cases that happen to look good proves little. Each case below is
the answer to a stated question, chosen by a fixed rule with ties broken by study id, and the set deliberately includes the cases
the system gets wrong, so the demo can show what the safeguards do and where they do not help. The rules are in `PROFILES`.

For each chosen study it writes demo/cases/<id>/image.jpg (a copy of the film) and an entry in demo/cases/manifest.json with the
indication, the labs on record and a short note for the person presenting. demo/ is git-ignored (MIMIC images and clinical text) and is
not part of any image; the API serves it only when it exists (app/api/examples.py).

Run from the repo root:  python scripts/build_demo_cases.py
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo" / "cases"
CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
SLUG = ["normal", "pneumonia", "lung_cancer"]


# ------------------------------------------------------------------ the rules
def _best(rows, key, reverse=True):
    return sorted(rows, key=lambda r: (key(r) * (-1 if reverse else 1), r["study_id"]))[0] if rows else None


PROFILES = [
    {"id": "clear-normal", "title": "A clear Normal", "question": "What does an easy, confident Normal look like?",
     "note": "All three inputs agree and the system is confident. Look at how the explanations support the call.",
     "pick": lambda r: r["true"] == "Normal" and r["pred"] == "Normal" and r["status"] == "agree" and not r["low"] and r["n_inputs"] == 3,
     "rank": lambda r: r["max_prob"]},
    {"id": "clear-pneumonia", "title": "A clear Pneumonia", "question": "What does an easy, confident Pneumonia look like?",
     "note": "All three inputs agree. The written note cites papers; each sentence passed an automated check against its source.",
     "pick": lambda r: r["true"] == "Pneumonia" and r["pred"] == "Pneumonia" and r["status"] == "agree" and not r["low"] and r["n_inputs"] == 3,
     "rank": lambda r: r["max_prob"]},
    {"id": "clear-lung-cancer", "title": "A Lung Cancer the system finds", "question": "Does it ever find the rare class with confidence?",
     "note": "The system names Lung Cancer. The rare class is the hardest one: this is the easy end of it.",
     "pick": lambda r: r["true"] == "Lung Cancer" and r["pred"] == "Lung Cancer" and not r["low"] and r["n_inputs"] == 3,
     "rank": lambda r: r["max_prob"]},
    {"id": "models-disagree-resolved", "title": "The three models disagree, and the pooled answer is right",
     "question": "What happens when the inputs conflict and the pooled answer turns out correct?",
     "note": "The consistency check reports a conflict between the models. Read what it says about how often such cases are right.",
     "pick": lambda r: r["status"] == "conflict" and r["n_inputs"] == 3 and r["pred"] == r["true"],
     "rank": lambda r: r["max_prob"]},
    {"id": "low-confidence", "title": "A case the system is not sure about", "question": "How does it behave when it should not be trusted?",
     "note": "The answer is headlined Uncertain with the best guess below it, and the prediction set lists what cannot be ruled out.",
     "pick": lambda r: r["low"] and r["n_inputs"] == 3,
     "rank": lambda r: -abs(r["max_prob"] - 0.47)},
    {"id": "no-labs", "title": "No blood tests on record", "question": "Does it still work when an input is missing?",
     "note": "Only the X-ray and the indication are used. The page says which inputs took part and the rules change for two inputs.",
     "pick": lambda r: not r["has_labs"] and r["pred"] == r["true"] and not r["low"],
     "rank": lambda r: r["max_prob"]},
    {"id": "wrong-but-flagged", "title": "Wrong, but the system says it is unsure", "question": "Do the safeguards catch a mistake?",
     "note": "The named class is not the true one, and the system flags low confidence. The prediction set still contains the true class.",
     "pick": lambda r: r["pred"] != r["true"] and r["low"] and r["true"] in r["set"],
     "rank": lambda r: -r["max_prob"]},
    {"id": "missed-cancer-alert-fires", "title": "A cancer the headline misses, and the alert catches",
     "question": "What is the separate cancer alert for?", "note": "The headline says something else, and the alert flags a possible Lung Cancer signal.",
     "pick": lambda r: r["true"] == "Lung Cancer" and r["pred"] != "Lung Cancer" and r["alert"],
     "rank": lambda r: r["p_cancer"]},
    {"id": "wrong-and-confident", "title": "Wrong, and the system does not know it", "question": "Where do the safeguards fail?",
     "note": "The named class is wrong and nothing flags it. This is shown on purpose: no confidence rule catches every error.",
     "pick": lambda r: r["pred"] != r["true"] and not r["low"] and r["status"] == "agree",
     "rank": lambda r: r["max_prob"]},
]


def select(records: list[dict]) -> list[dict]:
    """The chosen record for each profile that has at least one candidate, in profile order, each study used at most once."""
    used, chosen = set(), []
    for prof in PROFILES:
        candidates = [r for r in records if prof["pick"](r) and r["study_id"] not in used]
        best = _best(candidates, prof["rank"])
        if best is not None:
            used.add(best["study_id"])
            chosen.append({"profile": prof, "record": best, "candidates": len(candidates)})
    return chosen


# ------------------------------------------------------------------ the validation studies as the app would see them
def validation_records() -> list[dict]:
    from app.agent.decision import decide
    from app.agent.orchestrator import _case_from_split
    from app.agent.reasoning import reason
    from app.config import get_settings
    from app.pipelines.staged import provided_inputs

    cfg = get_settings()
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    cols = {"image": ("image", "image_p_"), "blood": ("blood", "blood_p_"), "text": ("text", "text_indication_p_")}
    stored = {m: pd.read_csv(ROOT / "experiments" / f"{d}_baseline" / "val_predictions.csv").set_index("study_id") for m, (d, _) in cols.items()}
    out = []
    for _, row in val.iterrows():
        case = _case_from_split(int(row.study_id), "val", set())
        have = provided_inputs(case["indication"], case["labs"] if case["labs"] else None, case["image_path"])
        r = reason({m: {c: float(stored[m].loc[row.study_id, cols[m][1] + s]) for c, s in zip(CLASSES, SLUG)} for m in have})
        dec = decide(r["belief"]["probabilities"], list(have), set_coverage=cfg.prediction_set_coverage, alert_recall=cfg.cancer_alert_recall)
        alert = (dec or {}).get("cancer_alert") or {}
        out.append({"study_id": int(row.study_id), "true": row.final_label, "pred": r["belief"]["predicted_class"],
                    "max_prob": float(max(r["belief"]["probabilities"].values())), "p_cancer": float(r["belief"]["probabilities"]["Lung Cancer"]),
                    "status": r["consistency"]["status"], "n_inputs": len(have), "inputs": sorted(have), "has_labs": "blood" in have,
                    "low": bool((dec or {}).get("low_confidence")), "set": (dec or {}).get("prediction_set") or [],
                    "alert": bool(alert.get("informative") and alert.get("flag")), "case": case})
    return out


def freeze(chosen: list[dict], out_dir: Path = DEMO) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for c in chosen:
        r, prof = c["record"], c["profile"]
        folder = out_dir / prof["id"]
        folder.mkdir(exist_ok=True)
        shutil.copyfile(r["case"]["image_path"], folder / "image.jpg")
        cases.append({"id": prof["id"], "title": prof["title"], "question": prof["question"], "note": prof["note"],
                      "indication": r["case"]["indication"] or "", "labs": {k: v for k, v in (r["case"]["labs"] or {}).items() if v is not None},
                      "expected": {"true_label": r["true"], "predicted_class": r["pred"], "pooled_confidence": round(r["max_prob"], 3),
                                   "consistency": r["status"], "inputs": r["inputs"], "low_confidence": r["low"],
                                   "prediction_set": r["set"], "cancer_alert": r["alert"]},
                      "study_id": r["study_id"], "candidates_for_this_profile": c["candidates"]})
    manifest = {"note": "Chosen by rule from validation studies by scripts/build_demo_cases.py. Contains MIMIC-derived text: do not commit.",
                "cases": cases}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = validation_records()
    chosen = select(records)
    manifest = freeze(chosen)
    print(f"{len(records)} validation studies scanned; {len(manifest['cases'])} cases frozen to {DEMO}")
    for c in manifest["cases"]:
        e = c["expected"]
        print(f"  {c['id']:28} true {e['true_label']:12} says {e['predicted_class']:12} conf {e['pooled_confidence']:.2f} {e['consistency']:9} "
              f"low {str(e['low_confidence']):5} alert {str(e['cancer_alert']):5} inputs {'+'.join(e['inputs'])}  ({c['candidates_for_this_profile']} candidates)")
    missing = [p["id"] for p in PROFILES if p["id"] not in {c["id"] for c in manifest["cases"]}]
    if missing:
        print("profiles with no candidate:", missing)


if __name__ == "__main__":
    main()
