"""
Feature-level (intermediate) multimodal fusion, as an ablation against the Phase 3 late
fusion (scripts/train_fusion.py, logpool, val macro F1 0.613).

Late fusion combines each modality's predicted probabilities. This instead concatenates
each modality's frozen representation into one vector and trains a small MLP end to end
on the concatenation:

  image: the 18 pathology probabilities from the frozen torchxrayvision backbone
         (experiments/image_baseline/feats_{train,val}.npy)
  blood: the 17 raw labs plus NLR and PLR (scripts/train_blood_baseline.py FEATURES),
         median-imputed with missingness indicators, since an MLP cannot take NaN the way
         the gradient-boosted blood model can
  text:  the 768-dim frozen Bio_ClinicalBERT embedding of the indication
         (experiments/text_baseline/emb_indication_{train,val}.npy), reduced by PCA so it
         does not swamp the other two blocks by sheer dimension

No out-of-fold step is needed here, unlike late fusion. Late fusion stacks each base
model's *predictions*, and those base models were fit on the train labels, so an OOF
scheme is required to keep the meta-model honest. Here the three blocks are frozen
encoder outputs that never saw our labels, so fitting one model directly on train and
evaluating once on val is already leakage-free.

Train/dev/val, not train/val: 15% of train is held out (subject-grouped, stratified) as a
dev set used only to pick the PCA dimension and to early-stop training. Val is touched
exactly once, for the number reported against logpool. This mirrors ordinary practice for
a single trained model with a hyperparameter to choose, rather than the stacking-specific
OOF scheme late fusion needed.

Also runs the same modality-subset ablation as late fusion, so the two fusion strategies
can be compared apples to apples.

Run from repo root (needs scripts/train_blood_baseline.py's FEATURES/add_derived, and the
cached embeddings from scripts/train_text_baseline.py and the image OOF kernel):
    python scripts/train_fusion_mlp.py
"""
from __future__ import annotations

# Windows: sklearn before torch can break torch's DLL init (WinError 1114). Import torch
# first (see CLAUDE.md section 4, gotcha 7).
import torch
import torch.nn as nn

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path("scripts").resolve()))
import train_blood_baseline as blood_mod  # noqa: E402  (reuses FEATURES / add_derived)

SEED = 42
OUT_DIR = Path("experiments/fusion_mlp")
LABEL_MAP = {"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]
LATE_FUSION_BAR = 0.613  # scripts/train_fusion.py, logpool, val macro F1

PCA_DIM_CANDIDATES = [16, 32, 64]
HIDDEN = 64          # same head width as the image baseline's 18->64->3 MLP
DROPOUT = 0.3
MAX_EPOCHS = 300
PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 128

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")  # small MLP on <=850 features; CPU is already fast


# ---------------------------------------------------------------- data loading
def load_frame(split: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/processed/{split}.csv")
    df = blood_mod.add_derived(df)
    df["y"] = df["final_label"].map(LABEL_MAP)
    return df


def fit_dev_split(df: pd.DataFrame) -> np.ndarray:
    """Subject-grouped stratified 85/15 split of train into fit/dev, same recipe as the
    late-fusion OOF folds (StratifiedGroupKFold, seed 42). fold==0 is dev (~15%)."""
    skf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)  # 1/7 ~ 14.3% dev
    fold = np.zeros(len(df), dtype=int)
    for i, (_, idx) in enumerate(skf.split(df, df["y"], groups=df["subject_id"])):
        fold[idx] = i
    return fold == 0  # boolean mask, True = dev


class Blocks:
    """Holds the three raw modality blocks for one split, before any fitted transform."""

    def __init__(self, df: pd.DataFrame, image_feats: np.ndarray, text_emb: np.ndarray):
        self.image = image_feats
        self.blood = df[blood_mod.FEATURES].to_numpy(dtype=float)
        self.text = text_emb
        self.y = df["y"].to_numpy()


def load_blocks(split: str) -> Blocks:
    df = load_frame(split)
    img = np.load(f"experiments/image_baseline/feats_{split}.npy")
    txt = np.load(f"experiments/text_baseline/emb_indication_{split}.npy")
    assert len(df) == len(img) == len(txt), f"{split}: row count mismatch across blocks"
    return Blocks(df, img, txt)


# ---------------------------------------------------------------- preprocessing
class Preprocessor:
    """Fits per-block transforms on one set of rows, applies to any other set. Keeps the
    three transforms in one place so "fit on fit-split, transform fit/dev/val" cannot
    accidentally leak by fitting on the wrong rows."""

    def __init__(self, modalities: tuple[str, ...], pca_dim: int | None):
        self.modalities = modalities
        self.pca_dim = pca_dim
        self.image_scaler = StandardScaler() if "image" in modalities else None
        self.blood_imputer = SimpleImputer(strategy="median", add_indicator=True) if "blood" in modalities else None
        self.blood_scaler = StandardScaler() if "blood" in modalities else None
        self.text_pca = PCA(n_components=pca_dim, random_state=SEED) if "text" in modalities else None
        self.text_scaler = StandardScaler() if "text" in modalities else None

    def fit(self, b: Blocks) -> "Preprocessor":
        if self.image_scaler is not None:
            self.image_scaler.fit(b.image)
        if self.blood_imputer is not None:
            imputed = self.blood_imputer.fit_transform(b.blood)
            self.blood_scaler.fit(imputed)
        if self.text_pca is not None:
            reduced = self.text_pca.fit_transform(b.text)
            self.text_scaler.fit(reduced)
        return self

    def transform(self, b: Blocks) -> np.ndarray:
        parts = []
        if "image" in self.modalities:
            parts.append(self.image_scaler.transform(b.image))
        if "blood" in self.modalities:
            parts.append(self.blood_scaler.transform(self.blood_imputer.transform(b.blood)))
        if "text" in self.modalities:
            parts.append(self.text_scaler.transform(self.text_pca.transform(b.text)))
        return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------- model
def make_mlp(in_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, 3),
    )


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = (1.0 / counts)
    w = w / w.sum() * 3
    return torch.tensor(w, dtype=torch.float32)


def train_mlp(X_fit, y_fit, X_dev, y_dev, verbose=False) -> tuple[nn.Module, float, int]:
    """Trains with early stopping on dev macro F1. Returns the best model (a deep copy of
    its state), that best dev macro F1, and the epoch it was reached at."""
    model = make_mlp(X_fit.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=class_weights(y_fit).to(DEVICE))

    Xt = torch.tensor(X_fit, device=DEVICE)
    yt = torch.tensor(y_fit, dtype=torch.long, device=DEVICE)
    Xd = torch.tensor(X_dev, device=DEVICE)
    n = len(Xt)

    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_pred = model(Xd).argmax(dim=1).cpu().numpy()
        dev_f1 = f1_score(y_dev, dev_pred, average="macro")
        if dev_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = dev_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch}: dev macro F1 {dev_f1:.4f} (best {best_f1:.4f} @ {best_epoch})")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_f1, best_epoch


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return model(torch.tensor(X, device=DEVICE)).argmax(dim=1).cpu().numpy()


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return torch.softmax(model(torch.tensor(X, device=DEVICE)), dim=1).cpu().numpy()


# ---------------------------------------------------------------- main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_b = load_blocks("train")
    val_b = load_blocks("val")
    dev_mask = fit_dev_split(load_frame("train"))
    fit_mask = ~dev_mask
    print(f"train n={len(train_b.y)} (fit {fit_mask.sum()}, dev {dev_mask.sum()}), val n={len(val_b.y)}")

    def split_blocks(b: Blocks, mask: np.ndarray) -> Blocks:
        out = Blocks.__new__(Blocks)
        out.image, out.blood, out.text, out.y = b.image[mask], b.blood[mask], b.text[mask], b.y[mask]
        return out

    fit_b, dev_b = split_blocks(train_b, fit_mask), split_blocks(train_b, dev_mask)

    # --- pick the PCA dimension for the text block using dev only, full 3-modality model ---
    print("\n=== choosing the text PCA dimension (dev set) ===")
    best_dim, best_dim_f1, scan = None, -1.0, {}
    for dim in PCA_DIM_CANDIDATES:
        prep = Preprocessor(("image", "blood", "text"), pca_dim=dim).fit(fit_b)
        X_fit, X_dev = prep.transform(fit_b), prep.transform(dev_b)
        _, dev_f1, ep = train_mlp(X_fit, fit_b.y, X_dev, dev_b.y)
        print(f"  text PCA dim {dim:3d}: dev macro F1 {dev_f1:.4f} (best epoch {ep})")
        scan[str(dim)] = round(float(dev_f1), 4)
        if dev_f1 > best_dim_f1:
            best_dim, best_dim_f1 = dim, dev_f1
    print(f"  chosen: PCA dim {best_dim} (dev macro F1 {best_dim_f1:.4f})")

    # --- final 3-modality model and the modality-subset ablation, all via the same fit/dev/val split ---
    results: dict = {
        "seed": SEED, "text_pca_dim": best_dim,
        "pca_scan_dev_macro_f1": dict(zip(map(str, PCA_DIM_CANDIDATES), [None] * len(PCA_DIM_CANDIDATES))),
        "n_fit": int(fit_mask.sum()), "n_dev": int(dev_mask.sum()), "n_val": len(val_b.y),
        "late_fusion_bar": LATE_FUSION_BAR, "ablation_val_macro_f1": {}, "ablation_dev_macro_f1": {},
    }

    mods = ("image", "blood", "text")
    combos = [c for r in (1, 2, 3) for c in itertools.combinations(mods, r)]
    final_model, final_prep = None, None
    for combo in combos:
        dim = best_dim if "text" in combo else None
        prep = Preprocessor(combo, pca_dim=dim).fit(fit_b)
        X_fit, X_dev, X_val = prep.transform(fit_b), prep.transform(dev_b), prep.transform(val_b)
        model, dev_f1, ep = train_mlp(X_fit, fit_b.y, X_dev, dev_b.y)
        val_pred = predict(model, X_val)
        val_f1 = f1_score(val_b.y, val_pred, average="macro")
        key = "+".join(combo)
        results["ablation_dev_macro_f1"][key] = round(float(dev_f1), 4)
        results["ablation_val_macro_f1"][key] = round(float(val_f1), 4)
        print(f"  {key:<18} dev {dev_f1:.4f}  val {val_f1:.4f}  (best epoch {ep}, {X_fit.shape[1]} features)")
        if combo == mods:
            final_model, final_prep, final_val_pred, final_val_f1 = model, prep, val_pred, val_f1

    print(f"\n=== final 3-modality MLP fusion (val) ===  macro F1 {final_val_f1:.4f}")
    print(classification_report(val_b.y, final_val_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))
    per_class = f1_score(val_b.y, final_val_pred, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(val_b.y, final_val_pred, labels=[0, 1, 2])
    results["pca_scan_dev_macro_f1"] = scan

    results["final"] = {
        "macro_f1": round(float(final_val_f1), 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(LABEL_NAMES, per_class)},
        "confusion_matrix": cm.tolist(),
        "n_features": sum([18, 19 + int(final_prep.blood_imputer.indicator_.features_.size) if final_prep.blood_imputer else 0, best_dim]),
    }
    results["verdict"] = (
        f"{results['final']['macro_f1']} vs late fusion (logpool) {LATE_FUSION_BAR} -> "
        + ("feature fusion HELPS" if results["final"]["macro_f1"] > LATE_FUSION_BAR + 1e-9 else "late fusion still wins")
    )

    torch.save(final_model.state_dict(), OUT_DIR / "fusion_mlp.pth")
    val_out = pd.DataFrame({
        "study_id": load_frame("val")["study_id"].to_numpy(),
        "final_label": load_frame("val")["final_label"].to_numpy(),
    })
    proba = predict_proba(final_model, final_prep.transform(val_b))
    for i, n in enumerate(LABEL_NAMES):
        val_out[f"mlpfusion_p_{n.replace(' ', '_').lower()}"] = proba[:, i]
    val_out["mlpfusion_pred"] = [LABEL_NAMES[i] for i in final_val_pred]
    val_out.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Feature-level (intermediate) fusion, MLP ablation",
        "",
        f"Train fit/dev/val: {results['n_fit']}/{results['n_dev']}/{results['n_val']}. "
        "Dev is used only to choose the text PCA dimension and to early-stop training; "
        "val is touched once, for the number below.",
        "",
        f"Text PCA dimension scan (dev macro F1): {scan}. Chosen: {best_dim}.",
        "",
        "## Result (val)",
        f"Macro F1 **{results['final']['macro_f1']}** "
        f"(Normal {results['final']['f1_per_class']['Normal']} / "
        f"Pneumonia {results['final']['f1_per_class']['Pneumonia']} / "
        f"Cancer {results['final']['f1_per_class']['Lung Cancer']}), "
        f"{results['final']['n_features']} input features.",
        "",
        f"**{results['verdict']}**",
        "",
        "## Modality-subset ablation (macro F1)",
        "| modalities | dev | val |",
        "|---|---|---|",
    ] + [
        f"| {k} | {results['ablation_dev_macro_f1'][k]} | {results['ablation_val_macro_f1'][k]} |"
        for k in results["ablation_val_macro_f1"]
    ] + [
        "",
        "## Comparison to late fusion (scripts/train_fusion.py)",
        f"Late fusion (logpool): {LATE_FUSION_BAR}. Feature fusion (this): {results['final']['macro_f1']}.",
        "",
        "Artifacts: fusion_mlp.pth, val_predictions.csv, metrics.json",
    ]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{results['verdict']}")
    print(f"saved -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
