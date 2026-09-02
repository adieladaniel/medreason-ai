"""
Grad-CAM + pathology attribution for the image model.

The image model is: frozen torchxrayvision DenseNet121 (weights
"densenet121-res224-mimic_ch") -> 18 CXR-pathology probabilities -> small MLP head
(18->64->3, experiments/image_baseline/image_head_final.pth).

Because our head consumes the 18 pathology *probabilities* (not conv features), a plain
Grad-CAM on "our head" has nothing spatial to attach to. So we keep the whole chain
differentiable -- image -> backbone conv trunk -> pathology probs -> our 3-class logit --
hook the backbone's final conv feature map, and backprop our predicted-class logit into
it. Two complementary outputs:

  1. heatmap  -- Grad-CAM over the backbone's last conv layer w.r.t. our predicted class
                 ("where on the film the model looked").
  2. pathology attribution -- d(our logit)/d(pathology prob) * (pathology prob), i.e.
     which of the 18 named pathology channels drove our prediction ("what it saw":
     e.g. Consolidation + Lung Opacity -> Pneumonia).

Preprocessing mirrors the training kernels exactly (grayscale, xrv.normalize to
~[-1024,1024], center-crop, resize 224) -- a mismatch here silently degrades the model.

Runs on CPU. First use downloads the backbone weights (~a few hundred MB) once.

CLI:  python -m explain.image --study-id 50262066 --out experiments/explain_samples/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from explain import LABEL_NAMES

HEAD_PATH = Path("experiments/image_baseline/image_head_final.pth")
BACKBONE_WEIGHTS = "densenet121-res224-mimic_ch"
IMG_SIZE = 224

_cache: dict = {}


def _load():
    if not _cache:
        import torchxrayvision as xrv

        backbone = xrv.models.DenseNet(weights=BACKBONE_WEIGHTS).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        head = nn.Sequential(
            nn.Linear(len(backbone.pathologies), 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3),
        )
        head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu", weights_only=True))
        head.eval()
        _cache.update(xrv=xrv, backbone=backbone, head=head,
                      pathologies=list(backbone.pathologies))
    return _cache


def _preprocess(image_path: str) -> torch.Tensor:
    from PIL import Image

    xrv = _load()["xrv"]
    from torchvision import transforms

    resize = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(IMG_SIZE)])
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, 255)
    arr = resize(arr[None, :, :])
    return torch.from_numpy(arr).unsqueeze(0)  # [1,1,224,224]


def _display_image(image_path: str) -> np.ndarray:
    from PIL import Image

    xrv = _load()["xrv"]
    from torchvision import transforms

    resize = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(IMG_SIZE)])
    img = Image.open(image_path).convert("L")
    arr = resize(np.asarray(img, dtype=np.float32)[None, :, :])[0]
    return (arr - arr.min()) / (arr.ptp() + 1e-8)


def explain_image(image_path: str, top_k: int = 6) -> dict:
    st = _load()
    backbone, head, pathologies = st["backbone"], st["head"], st["pathologies"]

    x = _preprocess(image_path)

    activations, grads = {}, {}
    conv = backbone.features  # final output is the [1,1024,7,7] pre-ReLU feature map

    def fwd_hook(_m, _i, out):
        activations["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    h = conv.register_forward_hook(fwd_hook)
    x = x.requires_grad_(True)  # backbone is frozen -> the input must carry grad for the
    try:                        # conv-activation gradients (Grad-CAM) to be computed
        probs18 = backbone(x)                     # [1,18], differentiable
        probs18.retain_grad()
        logits3 = head(probs18)                   # [1,3]
        pred = int(logits3.argmax(dim=1))
        proba = F.softmax(logits3, dim=1).detach()[0]
        backbone.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        logits3[0, pred].backward()
    finally:
        h.remove()

    A = activations["a"].detach()[0]              # [1024,7,7]
    G = grads["g"].detach()[0]                    # [1024,7,7]
    cam = F.relu((G.mean(dim=(1, 2), keepdim=True) * A).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                        align_corners=False)[0, 0].numpy()

    # pathology attribution: d(our logit)/d(prob) * prob, per named channel
    path_grad = probs18.grad.detach()[0].numpy()
    path_val = probs18.detach()[0].numpy()
    contrib = path_grad * path_val
    named = [(pathologies[i], float(path_val[i]), float(contrib[i]))
             for i in range(len(pathologies)) if pathologies[i]]
    is_normal = LABEL_NAMES[pred] == "Normal"
    # Normal call is explained by pathologies being weak -> rank by prob; disease call by
    # which channels push the predicted logit up -> rank by signed contribution.
    named.sort(key=(lambda t: -t[1]) if is_normal else (lambda t: -abs(t[2])))

    items = [
        {"feature": name, "value": round(val, 3), "effect": round(eff, 4),
         "direction": "supports" if eff > 0 else "against"}
        for name, val, eff in named[:top_k]
    ]
    if is_normal:
        considered = [f"{n} (p={v:.2f})" for n, v, _ in named[:3]]
        summary = (
            f"Image model predicts Normal (p={proba[pred]:.2f}). No pathology strongly "
            f"detected; closest signals judged not significant: {', '.join(considered)}."
        )
    else:
        drivers = [it["feature"] for it in items if it["direction"] == "supports"][:3]
        summary = (
            f"Image model predicts {LABEL_NAMES[pred]} (p={proba[pred]:.2f})"
            + (f", driven by internal channels: {', '.join(drivers)}." if drivers else ".")
        )
    return {
        "modality": "image",
        "predicted_class": LABEL_NAMES[pred],
        "probabilities": {n: round(float(p), 4) for n, p in zip(LABEL_NAMES, proba)},
        "attributions": {"method": "pathology-channel attribution (grad x value)", "items": items},
        "summary": summary,
        "_cam": cam,
        "_display": _display_image(image_path),
    }


def plot_image(res: dict, out_path: str | Path) -> Path:
    """`res` is the dict returned by explain_image() (must still have _cam, _display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, cam = res["_display"], res["_cam"]

    alpha = np.clip(cam, 0, 1) ** 1.5 * 0.7  # cold regions stay transparent

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("chest X-ray")
    axes[1].imshow(base, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=alpha)
    axes[1].set_title(f"Grad-CAM -> {res['predicted_class']}")
    for a in axes:
        a.axis("off")
    fig.suptitle(res["summary"], fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _image_from_split(study_id: int, split: str = "val") -> tuple[str, str]:
    import pandas as pd

    df = pd.read_csv(f"data/processed/{split}.csv")
    row = df.loc[df["study_id"] == study_id]
    if row.empty:
        raise SystemExit(f"study_id {study_id} not in {split}.csv")
    return row.iloc[0]["image_path"], row.iloc[0]["final_label"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int)
    ap.add_argument("--image", type=str)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None, help="dir or PNG path for the Grad-CAM figure")
    args = ap.parse_args()

    if args.image:
        image_path, true_label = args.image, "(n/a)"
    elif args.study_id:
        image_path, true_label = _image_from_split(args.study_id, args.split)
    else:
        raise SystemExit("pass --study-id or --image")

    res = explain_image(image_path)
    print(f"image: {image_path}\ntrue label: {true_label}")
    print(f"probabilities: {res['probabilities']}\n\n{res['summary']}\n")
    print(f"{'pathology channel':<28}{'value':>8}{'effect':>10}  direction")
    for it in res["attributions"]["items"]:
        print(f"{it['feature']:<28}{it['value']:>8}{it['effect']:>10}  {it['direction']}")
    if args.out:
        out = Path(args.out)
        if out.is_dir() or not out.suffix:
            out = out / f"image_{args.study_id or 'case'}.png"
        print(f"\nsaved plot -> {plot_image(res, out)}")
