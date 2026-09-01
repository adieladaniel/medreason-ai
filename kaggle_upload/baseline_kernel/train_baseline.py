"""
3-class (Normal/Pneumonia/Lung Cancer) chest X-ray classifier.
Uses torchxrayvision's DenseNet121 pretrained directly on MIMIC-CXR (weights
"densenet121-res224-mimic_ch") as a frozen feature extractor -- its own output
is a vector of CXR-pathology probabilities (Pneumonia, Lung Opacity, Lung Lesion,
Consolidation, etc.), which is a far more relevant feature space for this problem
than generic ImageNet features. A small trainable MLP head maps that vector to
our 3 classes. Class-weighted loss for the imbalance (Pneumonia 10165 / Normal
4277 / Cancer 1011). Runs on Kaggle's free GPU; not intended for local execution.
"""
import glob
import os
import random
import subprocess
import sys

# Kaggle's preinstalled PyTorch (cu128) dropped support for older GPU architectures
# (e.g. Tesla P100, compute capability 6.0) -- it only supports sm_70+. Pin an older
# build with broader architecture support so this runs regardless of which GPU
# (P100 or T4) Kaggle happens to assign.
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
from PIL import Image, ImageEnhance
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

print("=== /kaggle/input contents ===")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
    if depth <= 2:
        print(root, "-> dirs:", dirs, "files:", files[:5])

print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    # fail fast if this torch build still doesn't support the assigned GPU's
    # compute capability, rather than crashing later mid-training
    major, minor = torch.cuda.get_device_capability(0)
    test_tensor = torch.zeros(1, device=DEVICE)
    try:
        test_tensor + 1
        torch.cuda.synchronize()
        print(f"GPU compute capability sm_{major}{minor} confirmed working.")
    except RuntimeError as e:
        raise RuntimeError(
            f"GPU sm_{major}{minor} not supported by installed torch build ({torch.__version__}). "
            f"Try a different --accelerator / machine_shape, or an older torch/cu version."
        ) from e

processed_matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
print(f"train.csv matches: {processed_matches}")
PROCESSED_DIR = os.path.dirname(processed_matches[0])
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# the CXR image dataset's actual mount path can vary; locate it dynamically via metadata.csv
metadata_matches = glob.glob("/kaggle/input/**/metadata.csv", recursive=True)
print(f"metadata.csv matches: {metadata_matches}")
CXR_ROOT = os.path.dirname(metadata_matches[0])
print(f"CXR dataset root resolved to: {CXR_ROOT}")


def resolve_image_path(p):
    suffix = p.split("official_data_iccv_final/")[-1]
    return os.path.join(CXR_ROOT, "official_data_iccv_final", suffix)


LABEL_MAP = {"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]

IMG_SIZE = 224
xrv_resize = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(IMG_SIZE)])


class CXRDataset(Dataset):
    def __init__(self, df, augment):
        self.paths = df["image_path"].apply(resolve_image_path).tolist()
        self.labels = df["final_label"].map(LABEL_MAP).tolist()
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")  # xrv models expect single-channel input
        if self.augment:
            angle = random.uniform(-7, 7)
            img = img.rotate(angle, fillcolor=0)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        arr = np.asarray(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255)  # -> roughly [-1024, 1024], as xrv models expect
        arr = arr[None, :, :]
        arr = xrv_resize(arr)
        return torch.from_numpy(arr), self.labels[idx]


train_ds = CXRDataset(train_df, augment=True)
val_ds = CXRDataset(val_df, augment=False)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

class_counts = train_df["final_label"].map(LABEL_MAP).value_counts().sort_index()
weights = (1.0 / class_counts).values
weights = weights / weights.sum() * len(class_counts)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {dict(zip(LABEL_NAMES, weights))}")

# frozen MIMIC-CXR-pretrained backbone used purely as a feature extractor: its own
# output is a vector of CXR-pathology probabilities (Pneumonia, Lung Opacity,
# Lung Lesion, Consolidation, etc.) -- a domain-relevant feature space, not generic
# ImageNet features. Downloaded via internet (works now that the account is phone-verified).
backbone = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
backbone = backbone.to(DEVICE)
backbone.eval()
for param in backbone.parameters():
    param.requires_grad = False

NUM_PATHOLOGIES = len(backbone.pathologies)
print(f"Backbone pathologies ({NUM_PATHOLOGIES}): {backbone.pathologies}")

classifier_head = nn.Sequential(
    nn.Linear(NUM_PATHOLOGIES, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 3),
).to(DEVICE)

# sanity check the backbone on one real batch before committing to the full run
sanity_imgs, sanity_labels = next(iter(train_loader))
print(f"Sanity batch image tensor shape: {sanity_imgs.shape}, dtype: {sanity_imgs.dtype}")
with torch.no_grad():
    sanity_out = backbone(sanity_imgs.to(DEVICE))
print(f"Backbone output shape: {sanity_out.shape} (expected [batch, {NUM_PATHOLOGIES}])")
print(f"Backbone output range: min={sanity_out.min().item():.4f}, max={sanity_out.max().item():.4f}")
assert sanity_out.shape[1] == NUM_PATHOLOGIES, "backbone output dim mismatch with NUM_PATHOLOGIES"

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(classifier_head.parameters(), lr=1e-3)

EPOCHS = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_f1 = 0.0

for epoch in range(EPOCHS):
    classifier_head.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            pathology_probs = backbone(imgs)
        optimizer.zero_grad()
        outputs = classifier_head(pathology_probs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    scheduler.step()
    train_loss = running_loss / len(train_ds)

    classifier_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            pathology_probs = backbone(imgs)
            outputs = classifier_head(pathology_probs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_macro_f1: {val_f1:.4f} - lr: {scheduler.get_last_lr()}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(classifier_head.state_dict(), "/kaggle/working/best_head.pth")
        print(f"  -> saved new best head (f1={val_f1:.4f})")

print("\n=== Final classification report (best checkpoint) ===")
classifier_head.load_state_dict(torch.load("/kaggle/working/best_head.pth", weights_only=True))
classifier_head.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        pathology_probs = backbone(imgs)
        outputs = classifier_head(pathology_probs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
