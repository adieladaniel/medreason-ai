"""
Fit and measure the "does this look like a chest X-ray" guard (app/xray_guard.py).

Fitting uses training films only: 3,000 to fit the model of what a film looks like, and 1,000 different ones to set the distance
threshold (so that 99.5 percent of held-out training films pass). Validation films are then used only to measure. The sealed test
split is not opened.

Criteria, written before the first run:
  - at most 1 percent of validation films wrongly rejected;
  - at least 95 percent of each kind of clearly-not-an-X-ray image rejected: noise, flat colours, gradients, checkerboards,
    pages of text, shapes on white, smooth blobs, colour noise, colour gradients, and two natural colour photographs;
  - altered but real films (resized, re-compressed as JPEG at quality 30, saved as RGB, slightly rotated, blurred) should still pass
    at least 95 percent of the time.
Reported but not required, because they are hard for a gate this simple: an inverted film (negative), a film upside down, a
mosaic of one film tiled 3 by 3, and grayscale versions of the natural photographs (colour is the cue that catches real photographs).

History, kept because it matters: the first run (thumbnail distance and colour only) missed plain gradients entirely, 27 percent of
shapes on white and half of the grayscale photographs. Two cues were added, a minimum structure beyond a tilted plane and a maximum
share of saturated pixels, each set from held-out training films alone (half the 0.5th percentile of the structure measure; 1.25
times the 99.5th percentile of the saturated share). The negatives were drawn with seeds 43 and 99 while developing, so the final
numbers use a fresh seed (2024). Whether a hard case passes or fails is stated in the results, not hidden.

Run from the repo root:  python scripts/build_xray_guard.py
Writes app/data/xray_guard.json (aggregate statistics of thumbnails, no image) and experiments/xray_guard/results.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import xray_guard as g  # noqa: E402

SEED = 42
N_FIT, N_CAL = 3000, 1000
MAX_COLOURFULNESS = 12.0            # fixed in advance: grayscale films exported as RGB have equal channels
PASS_PERCENTILE = 99.5
OUT = Path("experiments/xray_guard")


def film(path: str) -> Image.Image:
    im = Image.open(path)
    im.load()
    return im


def thumbs_of(paths) -> np.ndarray:
    out = []
    for p in paths:
        im = Image.open(p)
        out.append(g.thumbnail(im))
    return np.stack(out)


# ------------------------------------------------------------------ things that are clearly not X-rays
def negatives(rng: np.random.Generator, n: int = 30) -> dict[str, list[Image.Image]]:
    def size():
        return int(rng.integers(256, 900)), int(rng.integers(256, 900))

    def gray(arr):
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")

    out: dict[str, list[Image.Image]] = {k: [] for k in (
        "gaussian noise", "uniform noise", "flat gray", "black", "white", "horizontal gradient", "vertical gradient", "checkerboard",
        "page of text", "shapes on white", "smooth blobs", "colour noise", "colour gradient", "natural colour photograph",
        "natural grayscale photograph (not required)")}
    for _ in range(n):
        w, h = size()
        out["gaussian noise"].append(gray(rng.normal(128, 40, (h, w))))
        out["uniform noise"].append(gray(rng.uniform(0, 255, (h, w))))
        out["flat gray"].append(gray(np.full((h, w), rng.integers(30, 220))))
        out["black"].append(gray(np.zeros((h, w))))
        out["white"].append(gray(np.full((h, w), 255)))
        out["horizontal gradient"].append(gray(np.tile(np.linspace(0, 255, w), (h, 1))))
        out["vertical gradient"].append(gray(np.tile(np.linspace(255, 0, h)[:, None], (1, w))))
        c = int(rng.integers(8, 64))
        out["checkerboard"].append(gray(((np.indices((h, w)).sum(axis=0) // c) % 2) * 255))
        page = Image.new("L", (w, h), 255)
        d = ImageDraw.Draw(page)
        for line in range(0, h, 14):
            d.text((10, line), "the quick brown fox jumps over the lazy dog " * 4, fill=0)
        out["page of text"].append(page)
        sh = Image.new("L", (w, h), 255)
        d = ImageDraw.Draw(sh)
        for _ in range(int(rng.integers(3, 9))):
            x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
            box = (x0, y0, x0 + int(rng.integers(20, w // 2)), y0 + int(rng.integers(20, h // 2)))
            (d.ellipse if rng.random() < 0.5 else d.rectangle)(box, fill=int(rng.integers(0, 255)))
        out["shapes on white"].append(sh)
        out["smooth blobs"].append(gray(rng.normal(128, 60, (h // 4, w // 4))).resize((w, h)).filter(ImageFilter.GaussianBlur(6)))
        out["colour noise"].append(Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB"))
        grad = np.stack([np.tile(np.linspace(0, 255, w), (h, 1)), np.tile(np.linspace(255, 0, h)[:, None], (1, w)),
                         np.full((h, w), rng.integers(0, 255))], axis=-1)
        out["colour gradient"].append(Image.fromarray(grad.astype(np.uint8), "RGB"))
    try:
        from sklearn.datasets import load_sample_images
        for photo in load_sample_images().images:                 # two colour photographs that ship with scikit-learn
            base = Image.fromarray(photo)
            for s in (0.3, 0.5, 0.8, 1.0, 1.3):
                out["natural colour photograph"].append(base.resize((int(base.width * s), int(base.height * s))))
                out["natural grayscale photograph (not required)"].append(
                    ImageOps.grayscale(base).resize((int(base.width * s), int(base.height * s))))
    except Exception as exc:                                       # noqa: BLE001
        print("natural photographs unavailable:", exc)
    return out


def alterations(im: Image.Image, rng: np.random.Generator) -> dict[str, Image.Image]:
    import io
    gray = im.convert("L")
    buf = io.BytesIO()
    gray.save(buf, "JPEG", quality=30)
    return {"resized to 512": gray.resize((512, int(512 * gray.height / gray.width))),
            "resized to 2000": gray.resize((2000, int(2000 * gray.height / gray.width))),
            "recompressed JPEG q30": Image.open(io.BytesIO(buf.getvalue())),
            "saved as RGB": gray.convert("RGB"),
            "rotated 3 degrees": gray.rotate(3, fillcolor=0),
            "blurred": gray.filter(ImageFilter.GaussianBlur(3))}


def hard_cases(im: Image.Image) -> dict[str, Image.Image]:
    gray = im.convert("L")
    small = gray.resize((300, 300))
    mosaic = Image.new("L", (900, 900))
    for i in range(3):
        for j in range(3):
            mosaic.paste(small, (i * 300, j * 300))
    return {"inverted film": ImageOps.invert(gray), "upside down": gray.rotate(180), "3 by 3 mosaic of one film": mosaic}


def main() -> None:
    rng = np.random.default_rng(SEED)
    train = pd.read_csv("data/processed/train.csv")              # the sealed split is never opened
    val = pd.read_csv("data/processed/val.csv")
    pick = rng.permutation(len(train))
    fit_paths = train.image_path.values[pick[:N_FIT]]
    cal_paths = train.image_path.values[pick[N_FIT:N_FIT + N_CAL]]

    print("thumbnails of training films ...", flush=True)
    tf, tc = thumbs_of(fit_paths), thumbs_of(cal_paths)
    extreme = np.array([g.extreme_fraction(Image.open(p)) for p in cal_paths])
    params = g.build_params(tf, tc, extreme, MAX_COLOURFULNESS, PASS_PERCENTILE)
    params["fitted_on"] = f"{N_FIT} training films, threshold from {N_CAL} other training films at the {PASS_PERCENTILE}th percentile"
    Path("app/data").mkdir(exist_ok=True)
    Path("app/data/xray_guard.json").write_text(json.dumps(params, separators=(",", ":")), encoding="utf-8")
    print(f"threshold {params['max_distance']:.2f}, min spread {params['min_spread']:.2f}, min structure {params['min_structure']:.1f}, "
          f"max extreme share {params['max_extreme_fraction']:.2f}", flush=True)

    def passes(im):
        return g.check(im, params)[0]

    results: dict = {"fit_films": N_FIT, "calibration_films": N_CAL, "max_distance": params["max_distance"],
                     "min_spread": params["min_spread"], "max_colourfulness": MAX_COLOURFULNESS}
    print("validation films ...", flush=True)
    ok = [passes(Image.open(p)) for p in val.image_path]
    results["validation_false_reject_rate"] = 1 - float(np.mean(ok))
    results["validation_films"] = len(ok)

    sample = val.sample(150, random_state=SEED).image_path.tolist()
    alt: dict[str, list[bool]] = {}
    hard: dict[str, list[bool]] = {}
    for p in sample:
        im = film(p)
        for k, v in alterations(im, rng).items():
            alt.setdefault(k, []).append(passes(v))
        for k, v in hard_cases(im).items():
            hard.setdefault(k, []).append(passes(v))
    results["altered_films_pass_rate"] = {k: float(np.mean(v)) for k, v in alt.items()}
    results["hard_cases_pass_rate_not_required"] = {k: float(np.mean(v)) for k, v in hard.items()}

    neg = negatives(np.random.default_rng(2024))
    results["not_an_xray_reject_rate"] = {k: 1 - float(np.mean([passes(i) for i in v])) for k, v in neg.items()}
    results["not_an_xray_counts"] = {k: len(v) for k, v in neg.items()}

    results["criteria_met"] = {
        "validation_false_reject_at_most_1_percent": results["validation_false_reject_rate"] <= 0.01,
        "every_negative_kind_rejected_at_least_95_percent": all(
            v >= 0.95 for k, v in results["not_an_xray_reject_rate"].items() if "not required" not in k),
        "every_alteration_passes_at_least_95_percent": all(v >= 0.95 for v in results["altered_films_pass_rate"].values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("validation_false_reject_rate", "altered_films_pass_rate", "not_an_xray_reject_rate",
                                              "hard_cases_pass_rate_not_required", "criteria_met")}, indent=1))


if __name__ == "__main__":
    main()
