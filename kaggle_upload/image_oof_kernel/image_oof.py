"""
Phase 3 prep -- generate the image modality's predictions for fusion.

The Phase 2 image baseline (v3: frozen torchxrayvision MIMIC-CXR DenseNet121 -> its 18
pathology-probability outputs -> small MLP head, val macro F1 0.50) only ever saved a
checkpoint on Kaggle. Fusion needs:
  1. out-of-fold (OOF) predictions on the *train* set -- so the fusion meta-model isn't
     fit on the image head's own training rows (standard stacking practice);
  2. predictions on the *val* set, dumped per study.

KEY SPEEDUP: the backbone is frozen, so each image's 18-dim feature vector is fixed.
Compute it once for all 11039 train + 2315 val images (~one pass through the backbone),
cache it, then every fold's head training is a few seconds on an [N, 18] matrix. This
turns what would be ~5x the original 35-min run into a single ~10-min run.

Deviation from the Phase 2 script: no train-time image augmentation (features are
precomputed once, deterministically). Augmentation gave only a marginal gain in Phase 2;
the point of this run is consistent fusion inputs, not squeezing the image model.

Folds: 5-fold StratifiedGroupKFold, seed 42, grouped by subject_id (a patient's studies
never split across folds) -- the same recipe scripts/generate_oof_predictions.py uses for
blood/text.

Outputs (to /kaggle/working, retrieve with `kaggle kernels output`):
  oof_train_predictions.csv, val_predictions.csv
    cols: subject_id, study_id, hadm_id, final_label, [fold,] image_p_normal,
          image_p_pneumonia, image_p_lung_cancer, image_pred
"""
import glob
import os
import subprocess
import sys

# Kaggle's preinstalled torch (cu128) dropped older GPU archs; pin a build that works on
# both P100 and T4 (see the Phase 2 kernel for the full story).
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch==2.4.1", "torchvision==0.19.1",
    "--index-url", "https://download.pytorch.org/whl/cu121",
], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "torchxrayvision"], check=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
N_FOLDS = 5
EPOCHS = 15
IMG_SIZE = 224
LABEL_MAP = {"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

processed = glob.glob("/kaggle/input/**/train.csv", recursive=True)
PROCESSED_DIR = os.path.dirname(processed[0])
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

CXR_ROOT = os.path.dirname(glob.glob("/kaggle/input/**/metadata.csv", recursive=True)[0])
print(f"CXR root: {CXR_ROOT}")


def resolve_image_path(p):
    suffix = p.split("official_data_iccv_final/")[-1]
    return os.path.join(CXR_ROOT, "official_data_iccv_final", suffix)


xrv_resize = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(IMG_SIZE)])


class CXRImages(Dataset):
    def __init__(self, df):
        self.paths = df["image_path"].apply(resolve_image_path).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255)
        arr = xrv_resize(arr[None, :, :])
        return torch.from_numpy(arr)


backbone = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch").to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False
N_PATH = len(backbone.pathologies)
print(f"Backbone pathologies ({N_PATH}): {backbone.pathologies}")


@torch.no_grad()
def extract(df):
    loader = DataLoader(CXRImages(df), batch_size=64, shuffle=False, num_workers=2)
    out = []
    for i, imgs in enumerate(loader):
        out.append(backbone(imgs.to(DEVICE)).cpu().numpy())
        if i % 20 == 0:
            print(f"  extracted {i * 64}/{len(df)}")
    return np.vstack(out).astype(np.float32)


print("Extracting frozen backbone features (once)...")
Xtr_full = extract(train_df)
Xva = extract(val_df)
ytr_full = train_df["final_label"].map(LABEL_MAP).values
yva = val_df["final_label"].map(LABEL_MAP).values
np.save("/kaggle/working/feats_train.npy", Xtr_full)
np.save("/kaggle/working/feats_val.npy", Xva)
print(f"Features: train {Xtr_full.shape}, val {Xva.shape}")


def make_head():
    return nn.Sequential(
        nn.Linear(N_PATH, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3),
    ).to(DEVICE)


def train_head(X_tr, y_tr, epochs=EPOCHS):
    counts = np.bincount(y_tr, minlength=3).astype(np.float64)
    w = (1.0 / counts)
    w = w / w.sum() * 3
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    head = make_head()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    n = len(Xt)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, 256):
            idx = perm[s : s + 256]
            opt.zero_grad()
            loss = crit(head(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    return head


@torch.no_grad()
def predict(head, X):
    head.eval()
    logits = head(torch.tensor(X, device=DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy()


# ---- OOF on train ----
skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold = np.full(len(train_df), -1)
oof = np.zeros((len(train_df), 3))
for k, (tr_idx, ho_idx) in enumerate(skf.split(train_df, ytr_full, groups=train_df["subject_id"])):
    fold[ho_idx] = k
    head = train_head(Xtr_full[tr_idx], ytr_full[tr_idx])
    oof[ho_idx] = predict(head, Xtr_full[ho_idx])
    f1 = f1_score(ytr_full[ho_idx], oof[ho_idx].argmax(1), average="macro")
    print(f"[image] fold {k}: n_heldout={len(ho_idx)}, macro F1={f1:.4f}")

print("\n=== OOF train classification report ===")
print(classification_report(ytr_full, oof.argmax(1), target_names=LABEL_NAMES, digits=3))

# ---- final head on all train -> val ----
final_head = train_head(Xtr_full, ytr_full)
val_proba = predict(final_head, Xva)
print("=== VAL classification report (final head) ===")
print(classification_report(yva, val_proba.argmax(1), target_names=LABEL_NAMES, digits=3))
print(f"VAL macro F1: {f1_score(yva, val_proba.argmax(1), average='macro'):.4f}")


def dump(df, proba, path, fold_col=None):
    out = df[["subject_id", "study_id", "hadm_id", "final_label"]].copy()
    if fold_col is not None:
        out["fold"] = fold_col
    for i, n in enumerate(LABEL_NAMES):
        out[f"image_p_{n.replace(' ', '_').lower()}"] = proba[:, i]
    out["image_pred"] = [LABEL_NAMES[i] for i in proba.argmax(1)]
    out.to_csv(path, index=False)
    print(f"saved {path}")


dump(train_df, oof, "/kaggle/working/oof_train_predictions.csv", fold_col=fold)
dump(val_df, val_proba, "/kaggle/working/val_predictions.csv")
torch.save(final_head.state_dict(), "/kaggle/working/image_head_final.pth")
print("done")
