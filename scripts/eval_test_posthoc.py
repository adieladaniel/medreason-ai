"""
POST-HOC, NOT IN THE PLAN: paired differences on the already-scored test predictions.

experiments/test_evaluation/PLAN.md promised each model alone and the pairs, "to check the ordering found on validation", but not a
paired interval for the gain from fusion. When the scored result showed the text model higher on test than on validation (macro F1
0.561 against 0.509), the fusion's advantage over the best single input looked smaller (+0.048 against +0.104 on validation), and the
unpaired intervals overlap. Whether the gain is real needs the paired comparison: both systems scored on the same resampled patients.

This reads the predictions and the labels the scoring step already used. It computes one more statistic from them and changes nothing: no
model, rule or threshold depends on it. It is reported as post-hoc, next to the same quantities on validation computed by the same code.

Run from the repo root, after --score:   python scripts/eval_test_posthoc.py
Writes experiments/test_evaluation/posthoc.json.
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_test_split as ev  # noqa: E402

B, SEED = 2000, 42
COMPARISONS = {                                                       # name: (the larger system, the smaller one)
    "all three minus text alone": (ev.MODS, ("text",)),
    "all three minus blood and text (what the image adds)": (ev.MODS, ("blood", "text")),
    "all three minus image and text (what blood adds)": (ev.MODS, ("image", "text")),
    "all three minus image and blood (what text adds)": (ev.MODS, ("image", "blood")),
}


def paired(P: dict, y: np.ndarray, subjects: np.ndarray, b: int = B, seed: int = SEED) -> dict:
    """Macro F1 of each pooled system on the same resampled patients, and the differences. The all-three pool of the validation
    convention (every model for every case), as the +0.104 on validation was computed."""
    pooled = {k: ev.pool(P, None, which) for k, which in {"all": ev.MODS, "text": ("text",), "image": ("image",), "blood": ("blood",),
                                                              "blood+text": ("blood", "text"), "image+text": ("image", "text"),
                                                              "image+blood": ("image", "blood")}.items()}
    pred = {k: v.argmax(1) for k, v in pooled.items()}
    uniq, inv = np.unique(subjects, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(b):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append({k: ev.macro_and_classes(y[idx], p[idx]).mean() for k, p in pred.items()})
    out = {"macro_f1": {k: float(ev.macro_and_classes(y, p).mean()) for k, p in pred.items()}, "differences": {}}
    key = {("text",): "text", ("blood", "text"): "blood+text", ("image", "text"): "image+text", ("image", "blood"): "image+blood"}
    for name, (_, small) in COMPARISONS.items():
        d = np.array([x["all"] - x[key[small]] for x in draws])
        point = out["macro_f1"]["all"] - out["macro_f1"][key[small]]
        out["differences"][name] = {"point": float(point), "interval": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not ev.RESULTS.exists():
        sys.exit("score the test split first (--score)")
    pred = pd.read_csv(ev.PREDICTIONS)
    labels = pd.read_csv(ev.TEST_CSV, usecols=["study_id", "final_label"])
    d = pred.merge(labels, on="study_id")
    y = d["final_label"].map({c: i for i, c in enumerate(ev.CLASSES)}).values
    P = {m: d[[f"{m}_p_{s}" for s in ev.SLUG]].values for m in ev.MODS}
    vP, _, vy, vsubjects = ev.validation_inputs()
    result = {"note": "post-hoc, not in PLAN.md; computed from the already-scored test predictions; nothing in the system depends on it",
              "test": paired(P, y, d["subject_id"].values), "validation": paired(vP, vy, vsubjects)}
    (ev.OUT / "posthoc.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{'':58}{'validation':>28}{'test':>28}")
    for name in COMPARISONS:
        v, t = result["validation"]["differences"][name], result["test"]["differences"][name]
        print(f"{name:58}{v['point']:>+10.3f} ({v['interval'][0]:+.3f} to {v['interval'][1]:+.3f}){t['point']:>+10.3f} ({t['interval'][0]:+.3f} to {t['interval'][1]:+.3f})")


if __name__ == "__main__":
    main()
