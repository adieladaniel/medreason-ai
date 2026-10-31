"""
Shared pieces of the version 2 experiment (experiments/v2_four_class/PLAN.md): loading the aligned inputs, fitting one modality's model with the
version 1 recipe, and the pooling rule. Nothing here reads a label of a validation or held-out study, and nothing here knows the test split.

Recipes are copied from version 1 and not tuned: blood is a HistGradientBoostingClassifier (scripts/generate_oof_predictions.py), text a
class-balanced logistic regression on the cached Bio_ClinicalBERT indication embedding, image the 18 -> 64 -> classes head with class-weighted
loss, Adam 1e-3, cosine schedule, 15 epochs (kaggle_upload/image_oof_kernel/image_oof.py). The image head is trained with three seeds and the
probabilities are averaged, because retraining that recipe moved the image macro F1 by about 0.03.
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_blood_baseline as blood_mod  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "v2_four_class"
PREDS = OUT / "preds"
CLASSES = ["Normal", "Pneumonia", "Lung Cancer", "Other"]
MODALITIES = ("blood", "text", "image")
SEED = 42
N_FOLDS = 5
IMAGE_SEEDS = (42, 43, 44)
EPOCHS = 15


# ------------------------------------------------------------------ inputs
def load_tables() -> dict:
    return {r: pd.read_csv(PROC / f"v2_{r}.csv") for r in ("train", "val", "heldout")}


def labels(table: pd.DataFrame) -> np.ndarray:
    return table["final_label"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(int)


def load_inputs(tables: dict) -> dict:
    """role -> {'blood': [n,19], 'text': [n,768], 'image': [n,18]}, rows in the order of the v2 tables. The alignment is checked, not assumed:
    in-scope rows come first and keep the version 1 order, then the Excluded rows in the order of their own tables, and the image outputs of
    the Excluded studies carry study ids that must match."""
    x18 = {"train": np.load(ROOT / "experiments" / "image_baseline" / "feats_train.npy"),
           "val": np.load(ROOT / "experiments" / "image_baseline" / "feats_val.npy")}
    emb_in = {"train": np.load(ROOT / "experiments" / "text_baseline" / "emb_indication_train.npy"),
              "val": np.load(ROOT / "experiments" / "text_baseline" / "emb_indication_val.npy")}
    excl_role = {"train": "train", "val": "val", "heldout": "none"}
    out = {}
    for role, table in tables.items():
        ex_mask = (table.source == "excluded").values
        n_in = int((~ex_mask).sum())
        assert not ex_mask[:n_in].any() and ex_mask[n_in:].all(), f"{role}: in-scope rows must come first"
        z = np.load(ROOT / "experiments" / "out_of_scope" / f"feats_{excl_role[role]}.npz")
        assert (z["study_id"] == table.study_id.values[n_in:]).all(), f"{role}: Excluded image outputs are not in table order"
        emb_ex = np.load(OUT / f"emb_indication_excluded_{excl_role[role]}.npy")
        assert len(emb_ex) == len(z["study_id"])
        if role == "heldout":
            image, text = z["xrv18"], emb_ex
        else:
            assert len(x18[role]) == n_in and len(emb_in[role]) == n_in
            image, text = np.vstack([x18[role], z["xrv18"]]), np.vstack([emb_in[role], emb_ex])
        blood = blood_mod.add_derived(table)[blood_mod.FEATURES].values.astype(np.float64)
        out[role] = {"blood": blood, "text": text.astype(np.float32), "image": image.astype(np.float32)}
    return out


# ------------------------------------------------------------------ one modality's model
def _hgb():
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, class_weight="balanced", random_state=SEED)


def _head(n_classes: int):
    return nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))


def train_head(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int):
    torch.manual_seed(seed)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    head = _head(n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    for _ in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            crit(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
    return head.eval()


@torch.no_grad()
def head_proba(head, X: np.ndarray) -> np.ndarray:
    return torch.softmax(head(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()


def fit_predict(modality: str, X_train: np.ndarray, y_train: np.ndarray, n_classes: int, X_eval: list) -> list:
    """Train one modality's model on (X_train, y_train) and return its probabilities for each matrix in X_eval. Image returns the three-seed mean;
    image_seeds() gives the individual seeds."""
    if modality == "blood":
        clf = _hgb().fit(X_train, y_train)
        return [clf.predict_proba(X) for X in X_eval]
    if modality == "text":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED).fit(X_train, y_train)
        return [clf.predict_proba(X) for X in X_eval]
    per_seed = fit_predict_image_seeds(X_train, y_train, n_classes, X_eval)
    return [np.mean([s[i] for s in per_seed], axis=0) for i in range(len(X_eval))]


def fit_predict_image_seeds(X_train, y_train, n_classes, X_eval) -> list:
    return [[head_proba(h, X) for X in X_eval] for h in (train_head(X_train, y_train, n_classes, s) for s in IMAGE_SEEDS)]


# ------------------------------------------------------------------ pooling
def logpool(P: np.ndarray) -> np.ndarray:
    """P is [n, modalities, classes]. The normalised geometric mean, the Phase 3 rule (scripts/eval_uncertainty.pooled)."""
    g = np.exp(np.log(P.clip(1e-6)).mean(axis=1))
    return g / g.sum(axis=1, keepdims=True)
