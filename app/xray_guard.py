"""
A cheap check that an uploaded image looks like a chest X-ray at all.

Why this exists: the API accepted any valid PNG or JPEG, so a photo of a cat or a screenshot got a Normal / Pneumonia / Lung Cancer
prediction with a straight face. The guard refuses what is clearly not a radiograph. It is a gate at the door, not a classifier
of films: it does not know whether the image is a chest, a frontal view, or the right way up, and it will pass a plain-looking
non-chest radiograph. Its job is to stop the absurd cases, and the docs say that plainly.

How, in three steps that each have a reason:
1. Colour. Radiographs are grayscale. A picture whose red, green and blue channels differ clearly is a photograph or a screenshot.
2. Flatness. A blank or near-uniform image has no structure to judge.
3. Structure. A smooth ramp (a plain gradient) is not a radiograph, and neither is a page that is mostly pure white or pure black.
   Real films vary a lot beyond a simple ramp across the picture, and almost never have more than about a third of their pixels at
   the extremes.
4. Shape of the picture. Reduce the image to a 20 by 20 grayscale thumbnail and ask how far it is from what chest X-rays look like,
   using a Gaussian model of the thumbnails of real training films (principal components, then a distance that counts both how
   unusual the main pattern is and how much of the picture the model cannot reproduce). Films share a recognisable layout (dark
   surround, two lung fields, a bright mediastinum), and noise, gradients, text and most photographs do not.

It uses no neural network and needs no download, and it does not depend on which image backbone the classifier uses. The numbers it
uses (mean, components, spreads, thresholds) are fitted by scripts/build_xray_guard.py and stored in app/data/xray_guard.json, which
holds only aggregate statistics of thumbnails and no image.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

PARAMS_PATH = Path(__file__).resolve().parent / "data" / "xray_guard.json"
THUMB = 20


def thumbnail(img: Image.Image) -> np.ndarray:
    """The 20 by 20 grayscale picture, flattened, in 0 to 255. JPEG decoding is done at reduced size, which is much faster
    for large films and makes no visible difference at this scale."""
    if img.format == "JPEG":
        img.draft("L", (THUMB * 4, THUMB * 4))
    return np.asarray(img.convert("L").resize((THUMB, THUMB), Image.BOX), dtype=np.float64).ravel()


def colourfulness(img: Image.Image) -> float:
    """Mean difference between colour channels, 0 to 255 (0 for any grayscale picture, whatever its mode)."""
    if img.mode in ("L", "LA", "1", "I", "I;16", "F"):
        return 0.0
    small = np.asarray(img.convert("RGB").resize((64, 64), Image.BOX), dtype=np.float64)
    return float((np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2])).mean() / 2)


def structure(t: np.ndarray) -> float:
    """How much the 20 by 20 thumbnail varies beyond the best-fitting tilted plane (a plain gradient scores about 0)."""
    yy, xx = np.mgrid[0:THUMB, 0:THUMB]
    a = np.c_[xx.ravel(), yy.ravel(), np.ones(THUMB * THUMB)]
    coef, *_ = np.linalg.lstsq(a, t, rcond=None)
    return float(np.std(t - a @ coef))


def extreme_fraction(img: Image.Image) -> float:
    """Share of pixels that are almost pure black or almost pure white, on a 64 by 64 grayscale reduction."""
    if img.format == "JPEG":
        img.draft("L", (256, 256))
    a = np.asarray(img.convert("L").resize((64, 64), Image.BOX), dtype=np.float64)
    return float(np.mean((a <= 2) | (a >= 253)))


def standardise(t: np.ndarray) -> np.ndarray:
    """Per-picture brightness and contrast removed, so exposure does not decide the answer, only the layout."""
    return (t - t.mean()) / t.std()


def fit(thumbs: np.ndarray, k: int = 20) -> dict:
    """thumbs: [n, 400] raw thumbnails of real films. Returns the model (without thresholds, which come from held-out films)."""
    z = np.stack([standardise(t) for t in thumbs])
    mean = z.mean(axis=0)
    u, s, vt = np.linalg.svd(z - mean, full_matrices=False)
    var = (s ** 2) / (len(z) - 1)
    return {"size": THUMB, "k": k, "mean": mean.tolist(), "components": vt[:k].tolist(), "variance": var[:k].tolist(),
            "residual_mean": None}


def build_params(fit_thumbs: np.ndarray, cal_thumbs: np.ndarray, cal_extreme: np.ndarray, max_colourfulness: float = 12.0,
                 pass_percentile: float = 99.5) -> dict:
    """The whole fitted guard. `fit_thumbs` fit the model of a film. Everything that decides rejection is set from other, held-out
    films (`cal_thumbs` and their saturated-pixel shares `cal_extreme`) so the guard is not graded on the films that defined it:
    the distance threshold passes `pass_percentile` percent of them, the structure floor is half the 0.5th percentile, the
    saturated-share ceiling is 1.25 times the 99.5th percentile (at least 0.10), and the flatness floor is half the 1st percentile of spread."""
    p = fit(fit_thumbs)
    z = np.stack([standardise(t) for t in fit_thumbs])
    c = (z - np.array(p["mean"])) @ np.array(p["components"]).T
    p["residual_mean"] = float((((z - (c @ np.array(p["components"]) + np.array(p["mean"]))) ** 2).sum(axis=1)).mean())
    p["max_distance"] = float(np.percentile(distances(cal_thumbs, p), pass_percentile))
    p["min_spread"] = float(np.percentile(fit_thumbs.std(axis=1), 1) / 2)
    p["max_colourfulness"] = max_colourfulness
    p["min_structure"] = float(np.percentile([structure(t) for t in cal_thumbs], 0.5) / 2)
    p["max_extreme_fraction"] = float(max(np.percentile(cal_extreme, 99.5) * 1.25, 0.10))        # never below 10 percent: a black border is normal
    return p


def distances(thumbs: np.ndarray, p: dict) -> np.ndarray:
    """How far each thumbnail is from the model: the larger of the main-pattern distance and the unexplained-picture distance,
    each scaled so that a typical film scores about 1."""
    z = np.stack([standardise(t) for t in thumbs])
    c = (z - np.array(p["mean"])) @ np.array(p["components"]).T
    mahal = (c ** 2 / np.array(p["variance"])).sum(axis=1) / p["k"]
    recon = c @ np.array(p["components"]) + np.array(p["mean"])
    resid = ((z - recon) ** 2).sum(axis=1) / p["residual_mean"]
    return np.maximum(mahal, resid)


@lru_cache(maxsize=1)
def load_params(path: str | None = None) -> dict | None:
    f = Path(path) if path else PARAMS_PATH
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def check(img: Image.Image, params: dict | None = None) -> tuple[bool, str | None, dict]:
    """(looks like a film, the reason it does not, the measurements). The reason is written for the person who uploaded it."""
    p = params if params is not None else load_params()
    if p is None:
        return True, None, {"guard": "not fitted"}                  # no fitted model: do not block anything on a guess
    col = colourfulness(img)
    t = thumbnail(img)
    spread = float(t.std())
    info = {"colourfulness": round(col, 2), "spread": round(spread, 2)}
    if col > p["max_colourfulness"]:
        return False, "this looks like a colour photograph or screenshot, and chest X-rays are grayscale", info
    if spread < p["min_spread"]:
        return False, "this image has no large-scale structure (it is blank, uniform or noise)", info
    if "min_structure" in p:
        st, ex = structure(t), extreme_fraction(img)
        info.update(structure=round(st, 2), extreme_fraction=round(ex, 3))
        if st < p["min_structure"]:
            return False, "this looks like a smooth gradient or a pattern with no anatomy in it", info
        if ex > p["max_extreme_fraction"]:
            return False, "this image is mostly pure black or pure white, unlike a radiograph", info
    d = float(distances(t[None, :], p)[0])
    info["distance"] = round(d, 2)
    if d > p["max_distance"]:
        return False, "this does not look like a chest X-ray (its overall layout is unlike a radiograph)", info
    return True, None, info
