"""
Does a different frozen image backbone make the image input useful for Lung Cancer? Validation split only.

The question, the arms and the adoption rule were written down before any feature was extracted, in
experiments/image_backbones/PLAN.md. The rule is encoded below in `adopt` so it cannot be reinterpreted after seeing numbers.

Every arm trains the same small head on the frozen backbone's outputs, with the current recipe (18 to 64 to 3, 15 epochs, Adam,
cosine, class-weighted loss, seed 42, no augmentation) and the same out-of-fold folds as the other modalities, then is scored
two ways: the image model alone, and fused with the stored blood and text models by the geometric-mean pool the app uses.
Intervals are cluster bootstrap over subjects (a patient has several studies), and differences from the control are paired: both
arms are scored on the same resampled patients.

Gate 0 first: the control, retrained here, must reproduce the stored image model and the stored fusion score, or nothing else is
read. The test split is never opened.

Run from the repo root, after scripts/extract_image_backbone_features.py:
    python scripts/eval_image_backbones.py
Writes experiments/image_backbones/results.json (aggregates) and, per arm, oof_train_predictions.csv and val_predictions.csv
(per study, git-ignored like the other prediction files).
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5
EPOCHS = 15
B = 2000
CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
SLUG = ["normal", "pneumonia", "lung_cancer"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
OUT = Path("experiments/image_backbones")

# backbone name -> where its cached outputs live ("mimic_ch" is the Kaggle-extracted set the app was built on)
FEATS = {"mimic_ch": "experiments/image_baseline/feats_{split}.npy",
         "nih": "experiments/image_backbones/feats_nih_{split}.npy",
         "pc": "experiments/image_backbones/feats_pc_{split}.npy"}
ARMS = {"control": ["mimic_ch"], "nih": ["nih"], "pc": ["pc"], "nih+mimic": ["nih", "mimic_ch"]}

# Gate 0 and the leakage-control readout, from PLAN.md
STORED_FUSED_F1, GATE_FUSED = 0.6133, 0.015
STORED_IMAGE_F1, GATE_IMAGE = 0.50, 0.03
LEAK_SAME, LEAK_LOWER = 0.03, 0.05


# ------------------------------------------------------------------ the rules written in PLAN.md
def adopt(arm: dict) -> tuple[bool, list[str]]:
    """The pre-registered adoption rule. `arm` holds: img_auc_cancer_lo (the arm's own lower bound), d_img_auc_cancer
    (point, lo), d_fused_f1 (point, lo), d_fused_auc_cancer (point). Returns (passes, the reasons it failed)."""
    why = []
    if not arm["d_img_auc_cancer"][1] > 0:
        why.append("image Lung Cancer AUC is not clearly above the control's (paired interval includes 0)")
    if not arm["img_auc_cancer_lo"] > 0.50:
        why.append("image Lung Cancer AUC interval still includes 0.50")
    if not arm["d_fused_f1"][0] >= 0:
        why.append("fused macro F1 is lower than the control's")
    if not arm["d_fused_f1"][1] >= -0.02:
        why.append("fused macro F1 may be more than 0.02 lower (paired interval)")
    if not arm["d_fused_auc_cancer"][0] >= 0:
        why.append("fused Lung Cancer AUC is lower than the control's")
    return (not why), why


def choose(passing: dict[str, dict]) -> str | None:
    """Among passing arms: highest fused macro F1; if two are within 0.005, the one needing fewer backbones."""
    if not passing:
        return None
    best = max(a["fused_f1"] for a in passing.values())
    close = {n: a for n, a in passing.items() if best - a["fused_f1"] <= 0.005}
    return min(close, key=lambda n: (len(ARMS[n]), -close[n]["fused_f1"]))


def leakage_readout(control_pneumonia_f1: float, arm_pneumonia_f1: float) -> str:
    d = arm_pneumonia_f1 - control_pneumonia_f1
    if d >= -LEAK_SAME:
        return "no evidence that overlap with the backbone's pretraining inflated the control's Pneumonia F1"
    if d <= -LEAK_LOWER:
        return "consistent with inflation, but a domain shift to MIMIC images is a competing explanation"
    return "inconclusive"


# ------------------------------------------------------------------ training, as in the Kaggle OOF kernel
def make_head(n_in: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))


def train_head(X: np.ndarray, y: np.ndarray, epochs: int = EPOCHS) -> nn.Module:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = 1.0 / counts
    w = w / w.sum() * 3
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    head = make_head(X.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            crit(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
    return head


@torch.no_grad()
def predict(head: nn.Module, X: np.ndarray) -> np.ndarray:
    head.eval()
    return torch.softmax(head(torch.tensor(X)), dim=1).numpy()


def features(names: list[str], split: str) -> np.ndarray:
    return np.hstack([np.load(FEATS[n].format(split=split)) for n in names]).astype(np.float32)


def run_arm(name: str, train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold train predictions and validation predictions of one arm."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Xtr, Xva = features(ARMS[name], "train"), features(ARMS[name], "val")
    ytr = train["final_label"].map(LABEL_MAP).values
    oof, fold = np.zeros((len(train), 3)), np.full(len(train), -1)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, ho) in enumerate(skf.split(train, ytr, groups=train["subject_id"])):
        fold[ho] = k
        oof[ho] = predict(train_head(Xtr[tr], ytr[tr]), Xtr[ho])
    final = train_head(Xtr, ytr)
    p_val = predict(final, Xva)
    OUT.joinpath(name.replace("+", "_")).mkdir(parents=True, exist_ok=True)
    for df, p, fn, f in ((train, oof, "oof_train_predictions.csv", fold), (val, p_val, "val_predictions.csv", None)):
        o = df[["subject_id", "study_id", "hadm_id", "final_label"]].copy()
        if f is not None:
            o["fold"] = f
        for i, s in enumerate(SLUG):
            o[f"image_p_{s}"] = p[:, i]
        o["image_pred"] = [CLASSES[i] for i in p.argmax(1)]
        o.to_csv(OUT / name.replace("+", "_") / fn, index=False)
    return oof, p_val


# ------------------------------------------------------------------ scoring
def pool(image_p: np.ndarray, blood_p: np.ndarray, text_p: np.ndarray) -> np.ndarray:
    """The app's rule (app/agent/belief.py): normalised geometric mean over the modalities."""
    s = np.stack([image_p, blood_p, text_p], axis=1).clip(1e-6)
    g = np.exp(np.log(s).mean(axis=1))
    return g / g.sum(axis=1, keepdims=True)


def auc(y: np.ndarray, p: np.ndarray, cls: int) -> float:
    return float(roc_auc_score(y == cls, p[:, cls])) if (y == cls).any() and (y != cls).any() else float("nan")


def stats(y: np.ndarray, img_p: np.ndarray, blood_p: np.ndarray, text_p: np.ndarray) -> dict:
    fused = pool(img_p, blood_p, text_p)
    f1_img = f1_score(y, img_p.argmax(1), average=None, labels=[0, 1, 2], zero_division=0)
    f1_fus = f1_score(y, fused.argmax(1), average=None, labels=[0, 1, 2], zero_division=0)
    return {"img_macro_f1": float(f1_img.mean()), "img_f1_pneumonia": float(f1_img[1]), "img_f1_cancer": float(f1_img[2]),
            "img_auc_cancer": auc(y, img_p, 2), "img_auc_pneumonia": auc(y, img_p, 1), "img_auc_normal": auc(y, img_p, 0),
            "fused_macro_f1": float(f1_fus.mean()), "fused_f1_cancer": float(f1_fus[2]), "fused_auc_cancer": auc(y, fused, 2)}


def interval(x: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    train, val = pd.read_csv("data/processed/train.csv"), pd.read_csv("data/processed/val.csv")   # only the two open splits, the sealed one is never named
    y = val["final_label"].map(LABEL_MAP).values

    def other(m: str) -> np.ndarray:
        col = {"blood": [f"blood_p_{s}" for s in SLUG], "text": [f"text_indication_p_{s}" for s in SLUG]}[m]
        d = pd.read_csv(f"experiments/{m}_baseline/val_predictions.csv").set_index("study_id").loc[val["study_id"]]
        return d[col].values
    blood_p, text_p = other("blood"), other("text")

    preds = {}
    for name in ARMS:
        if not all(Path(FEATS[n].format(split="val")).exists() for n in ARMS[name]):
            print(f"skipping {name}: its features have not been extracted")
            continue
        _, preds[name] = run_arm(name, train, val)
        s = stats(y, preds[name], blood_p, text_p)
        print(f"{name:10} image macro F1 {s['img_macro_f1']:.3f}  cancer AUC {s['img_auc_cancer']:.3f}  "
              f"fused macro F1 {s['fused_macro_f1']:.3f}", flush=True)

    # ---- Gate 0
    c = stats(y, preds["control"], blood_p, text_p)
    gate = {"fused_macro_f1": c["fused_macro_f1"], "stored_fused_macro_f1": STORED_FUSED_F1,
            "image_macro_f1": c["img_macro_f1"], "stored_image_macro_f1": STORED_IMAGE_F1}
    gate["passed"] = abs(c["fused_macro_f1"] - STORED_FUSED_F1) <= GATE_FUSED and abs(c["img_macro_f1"] - STORED_IMAGE_F1) <= GATE_IMAGE
    print("Gate 0 (control reproduces the stored models):", "PASSED" if gate["passed"] else "FAILED", gate)
    if not gate["passed"]:
        sys.exit("Gate 0 failed: find out why before reading the other arms (see PLAN.md).")

    # ---- paired cluster bootstrap
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(val["subject_id"].values, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    draws = {n: [] for n in preds}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for n in preds:
            draws[n].append(stats(y[idx], preds[n][idx], blood_p[idx], text_p[idx]))
    keys = list(draws["control"][0])
    arr = {n: {k: np.array([d[k] for d in draws[n]]) for k in keys} for n in preds}

    results = {"n_val": len(val), "n_val_cancer": int((y == 2).sum()), "bootstrap_draws": B, "gate0": gate, "arms": {}}
    passing = {}
    for n in preds:
        pt = stats(y, preds[n], blood_p, text_p)
        row = {"backbones": ARMS[n], "point": pt, "interval": {k: interval(arr[n][k]) for k in keys}}
        if n != "control":
            row["paired_difference_vs_control"] = {
                k: {"point": pt[k] - c[k], "interval": interval(arr[n][k] - arr["control"][k])}
                for k in ("img_auc_cancer", "img_macro_f1", "img_f1_pneumonia", "fused_macro_f1", "fused_auc_cancer", "fused_f1_cancer")}
            d = row["paired_difference_vs_control"]
            ok, why = adopt({"img_auc_cancer_lo": row["interval"]["img_auc_cancer"][0],
                             "d_img_auc_cancer": (d["img_auc_cancer"]["point"], d["img_auc_cancer"]["interval"][0]),
                             "d_fused_f1": (d["fused_macro_f1"]["point"], d["fused_macro_f1"]["interval"][0]),
                             "d_fused_auc_cancer": (d["fused_auc_cancer"]["point"],)})
            row["passes_adoption_rule"], row["failed_because"] = ok, why
            if ok:
                passing[n] = {"fused_f1": pt["fused_macro_f1"]}
            if n in ("nih", "pc"):
                row["leakage_readout"] = leakage_readout(c["img_f1_pneumonia"], pt["img_f1_pneumonia"])
        results["arms"][n] = row
    results["decision"] = choose(passing) or "keep the current image model"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    def ci(v, i):
        return f"{v:.3f} ({i[0]:.3f} to {i[1]:.3f})"
    print("\nimage alone: Lung Cancer AUC | Pneumonia F1 | macro F1      fused: macro F1 | Lung Cancer AUC")
    for n, r in results["arms"].items():
        p, i = r["point"], r["interval"]
        print(f"{n:10} {ci(p['img_auc_cancer'], i['img_auc_cancer'])} | {p['img_f1_pneumonia']:.3f} | {p['img_macro_f1']:.3f}"
              f"      {ci(p['fused_macro_f1'], i['fused_macro_f1'])} | {ci(p['fused_auc_cancer'], i['fused_auc_cancer'])}")
        if "passes_adoption_rule" in r:
            print(f"           adoption rule: {'PASS' if r['passes_adoption_rule'] else 'fail: ' + '; '.join(r['failed_because'])}")
        if "leakage_readout" in r:
            print(f"           leakage readout: {r['leakage_readout']}")
    print("\nDECISION:", results["decision"])


if __name__ == "__main__":
    main()
