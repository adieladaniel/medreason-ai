"""
Extract the frozen outputs of other torchxrayvision backbones for every train and validation study.

Why this exists: experiments/image_backbones/PLAN.md asks whether a backbone with trained Nodule and Mass outputs (and no
MIMIC-CXR pretraining) makes the image input useful for Lung Cancer. The backbone is frozen, so its 18 outputs for an image
never change; extract them once and every head trained on top of them is a few seconds of work.

The preprocessing is the app's (explain/image.py: grayscale, xrv normalise, centre crop, resize to 224), and a check on 64
images showed it reproduces the cached Kaggle features of the current backbone to 0.000002. The images are decoded once and
passed through every requested backbone, because decoding is about a third of the time of one backbone.

Only the train and validation splits are read. The test split stays sealed, so this script has no way to ask for it.

Resumable: each chunk of images is saved as it finishes and skipped on the next run. Run from the repo root:
    python scripts/extract_image_backbone_features.py --weights nih pc
Output: experiments/image_backbones/feats_<name>_<split>.npy (git-ignored: derived from MIMIC images), in the row order of
data/processed/<split>.csv.
"""
import torch  # noqa: F401  (Windows: import torch before sklearn and shap, or torch's DLL init can fail)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torchxrayvision as xrv
from PIL import Image
from torchvision import transforms

OUT = Path("experiments/image_backbones")
SPLITS = ("val", "train")                      # the sealed split is deliberately not an option
CHUNK = 512
IMG_SIZE = 224
_resize = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(IMG_SIZE)])


def preprocess(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return torch.from_numpy(_resize(xrv.datasets.normalize(arr, 255)[None]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="short names such as nih pc mimic_ch")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    models = {w: xrv.models.DenseNet(weights=f"densenet121-res224-{w}").eval() for w in args.weights}
    (OUT / "_chunks").mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        paths = pd.read_csv(f"data/processed/{split}.csv")["image_path"].tolist()
        starts = list(range(0, len(paths), CHUNK))
        t0 = time.perf_counter()
        for k, s in enumerate(starts):
            f = OUT / "_chunks" / f"{split}_{k:03d}_{'-'.join(args.weights)}.npz"
            if f.exists():
                continue
            batch = torch.stack([preprocess(p) for p in paths[s:s + CHUNK]])
            with torch.no_grad():
                out = {w: np.vstack([m(batch[i:i + 64]).numpy() for i in range(0, len(batch), 64)]).astype(np.float32)
                       for w, m in models.items()}
            np.savez(f, **out)
            done = min(s + CHUNK, len(paths))
            print(f"{split}: {done}/{len(paths)} images, {time.perf_counter() - t0:.0f} s", flush=True)
        for w in args.weights:
            parts = [np.load(OUT / "_chunks" / f"{split}_{k:03d}_{'-'.join(args.weights)}.npz")[w] for k in range(len(starts))]
            arr = np.vstack(parts)
            assert len(arr) == len(paths), (w, split, arr.shape, len(paths))
            np.save(OUT / f"feats_{w}_{split}.npy", arr)
            print(f"saved feats_{w}_{split}.npy {arr.shape}")


if __name__ == "__main__":
    main()
