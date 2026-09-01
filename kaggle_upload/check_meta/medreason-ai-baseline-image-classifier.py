"""
Baseline 3-class (Normal/Pneumonia/Lung Cancer) chest X-ray classifier.
Frozen ImageNet-pretrained DenseNet121 backbone + trainable linear head,
class-weighted loss to handle the imbalance (Pneumonia 10165 / Normal 4277 / Cancer 1011).
Runs on Kaggle's free GPU; not intended for local execution.
"""
import glob
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# no horizontal flip: it would reverse clinically meaningful heart/organ position
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class CXRDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["image_path"].apply(resolve_image_path).tolist()
        self.labels = df["final_label"].map(LABEL_MAP).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


train_ds = CXRDataset(train_df, train_tf)
val_ds = CXRDataset(val_df, eval_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

class_counts = train_df["final_label"].map(LABEL_MAP).value_counts().sort_index()
weights = (1.0 / class_counts).values
weights = weights / weights.sum() * len(class_counts)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {dict(zip(LABEL_NAMES, weights))}")

weights_matches = glob.glob("/kaggle/input/**/densenet121_imagenet.pth", recursive=True)
print(f"pretrained weights matches: {weights_matches}")
model = models.densenet121(weights=None)
model.load_state_dict(torch.load(weights_matches[0], map_location=DEVICE))

for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(model.classifier.in_features, 3)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

EPOCHS = 8
best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_ds)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_macro_f1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        print(f"  -> saved new best model (f1={val_f1:.4f})")

print("\n=== Final classification report (best checkpoint) ===")
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
