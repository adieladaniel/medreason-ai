"""
Step 0 of the version 2 plan (experiments/v2_four_class/PLAN.md): checks that could still change the plan, run before any training.

1. Are the cached image backbone outputs for the in-scope training and validation studies really aligned with the tables? They are saved
   without study ids, so a row-order mistake would silently poison every later step. The first rows of each table are recomputed from the film
   and compared. The Excluded outputs carry study ids and are compared with their tables directly.
2. What is in the Excluded class? Its "No Finding" and tuberculosis-code composition caps what a film or text model can do, so it is
   measured before the arms are trained.
3. Hashes of the version 1 system and of the data tables, so that "version 1 did not change" can be checked at the end.
4. How long does the one slow step take on the CPU: embedding the Excluded studies' indication text with Bio_ClinicalBERT?

Reads no label of any in-scope test study and no test table. Writes experiments/v2_four_class/step0.json (aggregates and hashes only, no ids).
Run from the repo root:  python scripts/v2_step0.py
"""
import torch  # noqa: F401  (Windows: torch before sklearn, see CLAUDE.md gotcha 7)

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "experiments" / "v2_four_class"
ROLES = ("train", "val", "none")                      # the sealed test patients are deliberately not an option
SPOT = 6                                              # films recomputed per table
TIMED = 128                                           # indications embedded to time the slow step
DATA_FILES = ["data/processed/train.csv", "data/processed/val.csv", "data/processed/excluded_train.csv",
              "data/processed/excluded_val.csv", "data/processed/excluded_none.csv", "data/interim/final_study_labels.csv",
              "experiments/image_baseline/feats_train.npy", "experiments/image_baseline/feats_val.npy",
              "experiments/out_of_scope/feats_train.npz", "experiments/out_of_scope/feats_val.npz",
              "experiments/out_of_scope/feats_none.npz", "experiments/fusion/val_predictions.csv"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_image_features() -> dict:
    """Cached backbone outputs against a fresh pass over the first rows of each table."""
    import torch.nn.functional as F  # noqa: F401

    import explain.image as image
    st = image._load()
    out = {}
    for name in ("train", "val"):
        table = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
        cached = np.load(ROOT / "experiments" / "image_baseline" / f"feats_{name}.npy")
        diffs = []
        for i in range(SPOT):
            with torch.no_grad():
                fresh = st["backbone"](image._preprocess(table.image_path.iloc[i]))[0].numpy()
            diffs.append(float(np.abs(fresh - cached[i]).max()))
        out[name] = {"rows": int(len(table)), "cached_shape": list(cached.shape), "rows_match": bool(len(table) == len(cached)),
                     "recomputed": SPOT, "largest_difference": max(diffs)}
    for role in ROLES:
        table = pd.read_csv(ROOT / "data" / "processed" / f"excluded_{role}.csv")
        z = np.load(ROOT / "experiments" / "out_of_scope" / f"feats_{role}.npz")
        out[f"excluded_{role}"] = {"rows": int(len(table)), "cached_shape": list(z["xrv18"].shape),
                                   "study_ids_in_order": bool((z["study_id"] == table.study_id.values).all()),
                                   "has_blood_share": round(float(z["has_blood"].mean()), 3),
                                   "has_text_share": round(float(z["has_text"].mean()), 3)}
    return out


def excluded_composition() -> dict:
    """What the Excluded class is made of, from the labelling tables (no report text is read)."""
    labels = pd.read_csv(ROOT / "data" / "interim" / "final_study_labels.csv")
    for c in ("no_finding", "pneumonia_mentioned", "cancer_mentioned"):
        labels[c] = labels[c].astype(str).str.lower().eq("true")
    out = {}
    for role in ROLES:
        ex = pd.read_csv(ROOT / "data" / "processed" / f"excluded_{role}.csv", usecols=["study_id", "subject_id"])
        m = ex.merge(labels, on="study_id", how="left", suffixes=("", "_l"))
        assert len(m) == len(ex) and m.final_label.eq("Excluded").all()
        n = len(m)
        out[role] = {
            "studies": n, "patients": int(ex.subject_id.nunique()),
            "report_says_no_finding": round(float(m.no_finding.mean()), 3),
            "tb_icd_code": round(float((m.icd_tb == 1).mean()), 3),
            "no_finding_and_tb_code": round(float((m.no_finding & (m.icd_tb == 1)).mean()), 3),
            "has_findings_other_than_the_two": round(float((~m.no_finding).mean()), 3),
            "pneumonia_or_cancer_icd_but_excluded": int(((m.icd_pneumonia == 1) | (m.icd_cancer == 1)).sum()),
        }
    return out


def hashes() -> dict:
    from eval_test_split import frozen_system
    return {"version1_system": frozen_system(), "data": {f: sha(ROOT / f) for f in DATA_FILES}}


def time_embedding() -> dict:
    """Embed TIMED Excluded indications exactly as scripts/train_text_baseline.py does, and extrapolate to all Excluded studies."""
    from transformers import AutoModel, AutoTokenizer

    import train_text_baseline as tb
    table = pd.read_csv(ROOT / "data" / "processed" / "excluded_val.csv").head(TIMED)
    texts = [tb.extract_indication(tb.read_report(p)) for p in table.report_path]
    tok = AutoTokenizer.from_pretrained(tb.MODEL_NAME)
    model = AutoModel.from_pretrained(tb.MODEL_NAME).eval()
    t0 = time.time()
    emb = tb.embed(texts, tok, model)
    secs = time.time() - t0
    total = sum(len(pd.read_csv(ROOT / "data" / "processed" / f"excluded_{r}.csv", usecols=["study_id"])) for r in ROLES)
    words = [len(t.split()) for t in texts]
    return {"timed": TIMED, "seconds": round(secs, 1), "seconds_per_text": round(secs / TIMED, 4), "excluded_total": total,
            "estimated_minutes_for_all": round(secs / TIMED * total / 60, 1), "embedding_dim": int(emb.shape[1]),
            "empty_indications_in_timed": int(sum(w == 0 for w in words)), "median_words": float(np.median(words)),
            "threads": torch.get_num_threads(), "cuda": bool(torch.cuda.is_available())}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"image_features": check_image_features()}
    print(json.dumps(result["image_features"], indent=1))
    result["excluded_composition"] = excluded_composition()
    print(json.dumps(result["excluded_composition"], indent=1))
    result["text_embedding_timing"] = time_embedding()
    print(json.dumps(result["text_embedding_timing"], indent=1))
    result["hashes"] = hashes()
    (OUT / "step0.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT / "step0.json")


if __name__ == "__main__":
    main()
