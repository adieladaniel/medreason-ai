"""
Features for the out-of-scope detector (experiments/out_of_scope/PLAN.md), for Excluded studies: the image backbone's 18 pathology outputs, the
three models' class probabilities, and which inputs the study really had. The same quantities for the in-scope training and validation studies
already exist (the cached backbone outputs and the out-of-fold and validation prediction files), so only the Excluded tables need this.

For each study: one pass through the frozen backbone (its 18 outputs), the image head on those, the blood model (labs, or all-missing when there
are none, as in the validation convention), and the text model on the indication (an empty string when there is none), plus flags for which
inputs were present by the app's rule (app.pipelines.staged.provided_inputs).

Resumable: each chunk is saved as it finishes. Only the roles named in PLAN.md can be asked for; the sealed test split cannot.
Run from the repo root, after scripts/build_final_dataset.py --excluded --role <role>:
    python scripts/extract_scope_features.py --roles train val none
Output: experiments/out_of_scope/feats_<role>.npz (git-ignored: derived from MIMIC).
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "experiments" / "out_of_scope"
ROLES = ("train", "val", "none")                      # the sealed test patients are deliberately not an option
CHUNK = 250
SLUG = ["normal", "pneumonia", "lung_cancer"]


def study_features(row: pd.Series, st: dict, predictors: dict) -> dict:
    """Everything the detector uses for one study."""
    import torch.nn.functional as F

    import explain.image as image
    from eval_test_split import case_inputs
    from app.pipelines.staged import provided_inputs

    c = case_inputs(row)
    have = provided_inputs(c["indication"], c["labs"], c["image"])
    with torch.no_grad():
        f18 = st["backbone"](image._preprocess(c["image"]))
        p_img = F.softmax(st["head"](f18), dim=1)[0].numpy()
    blood = predictors["blood"](c["labs"])["probabilities"]
    text = predictors["text"](c["indication"] or "")["probabilities"]
    cls = ["Normal", "Pneumonia", "Lung Cancer"]
    return {"study_id": int(row["study_id"]), "subject_id": int(row["subject_id"]), "xrv18": f18[0].numpy().astype(np.float32),
            "image_p": p_img.astype(np.float32), "blood_p": np.array([blood[k] for k in cls], np.float32),
            "text_p": np.array([text[k] for k in cls], np.float32), "has_blood": "blood" in have, "has_text": "text" in have}


def stack(records: list[dict]) -> dict:
    return {k: np.stack([r[k] for r in records]) if not np.isscalar(records[0][k]) else np.array([r[k] for r in records])
            for k in records[0]}


def run_role(role: str) -> None:
    import explain.image as image
    from app.agent.tools import PREDICTORS

    table = pd.read_csv(ROOT / "data" / "processed" / f"excluded_{role}.csv")
    st = image._load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_chunks").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for k, s in enumerate(range(0, len(table), CHUNK)):
        f = OUT / "_chunks" / f"{role}_{k:03d}.npz"
        if f.exists():
            continue
        recs = [study_features(r, st, PREDICTORS) for _, r in table.iloc[s:s + CHUNK].iterrows()]
        np.savez(f, **stack(recs))
        print(f"{role}: {min(s + CHUNK, len(table))}/{len(table)} studies, {time.perf_counter() - t0:.0f} s", flush=True)
    parts = [np.load(OUT / "_chunks" / f"{role}_{k:03d}.npz") for k in range(-(-len(table) // CHUNK))]
    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    assert len(merged["study_id"]) == len(table)
    np.savez(OUT / f"feats_{role}.npz", **merged)
    print(f"saved feats_{role}.npz ({len(table)} studies)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    args = ap.parse_args()
    torch.set_num_threads(10)
    for role in args.roles:
        run_role(role)


if __name__ == "__main__":
    main()
