"""Fetch the public-domain chest film used for the public demo recording, make it acceptable to the app, and write its credit line.

Why: the nine validation demo cases show real MIMIC-CXR films and MIMIC clinical text. The PhysioNet license for that data says the
licensee "will not share access to PhysioNet restricted data with anyone else" and limits use to "lawful use in scientific research and no
other", and it does not say whether a recording that shows the data is allowed. So nothing public (a video, a screenshot, a post) should
show a MIMIC film or MIMIC text. A public demo needs a film that is not from MIMIC, and this is one: a posteroanterior chest radiograph
from the CDC Public Health Image Library (image 17298, credit "CDC/ Dr. Jack Poland"), which the library lists as public domain, with
credit requested. docs/DEMO_SCRIPT.md has the whole recording plan.

What it does: downloads the library's low-resolution JPEG (the high-resolution file is behind a page postback), converts it to a single
grayscale channel (the JPEG carries a slight colour tint, and the app's non-X-ray guard refuses colour pictures, which is right for
photographs and wrong for this tinted film), checks the result with the same guard the server uses, and writes
`data/demo_public/chest_pa_cdc17298.png` and `CREDIT.txt`. `data/` is git-ignored. It needs the network once.

  python scripts/prepare_public_demo.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "demo_public"
SOURCE_PAGE = "https://wwwn.cdc.gov/phil/Details.aspx?pid=17298"
IMAGE_URL = "https://wwwn.cdc.gov/phil///PHIL_Images/17298/17298_lores.jpg"
CREDIT = ("Posteroanterior chest radiograph, CDC Public Health Image Library image 17298, 1975. Credit: CDC/ Dr. Jack Poland. "
          "Public domain; the library asks that CDC be credited as the source. Converted to one grayscale channel for this demo.\n"
          f"Source: {SOURCE_PAGE}\n")


def to_grayscale(image: Image.Image) -> Image.Image:
    """The same picture on one channel. Nothing is cropped, scaled or enhanced."""
    return image.convert("L")


def prepare(raw: bytes | Path, out_dir: Path = OUT_DIR) -> tuple[Path, bool, str | None]:
    """Write the converted film and its credit into out_dir. Returns (film path, whether the app's guard accepts it, the guard's reason)."""
    from app import xray_guard
    import io
    image = Image.open(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    gray = to_grayscale(image)
    out_dir.mkdir(parents=True, exist_ok=True)
    film = out_dir / "chest_pa_cdc17298.png"
    gray.save(film, "PNG")
    (out_dir / "CREDIT.txt").write_text(CREDIT, encoding="utf-8")
    accepted, reason = xray_guard.check(gray)[:2]
    return film, bool(accepted), reason


def main() -> None:
    with urllib.request.urlopen(IMAGE_URL, timeout=30) as resp:
        raw = resp.read()
    film, accepted, reason = prepare(raw)
    print(f"wrote {film} ({len(raw)} bytes downloaded) and CREDIT.txt next to it")
    print("the app's guard accepts it" if accepted else f"the app's guard REFUSES it: {reason}")
    print("Show the credit line in the video or its description:", CREDIT.splitlines()[0])
    sys.exit(0 if accepted else 1)


if __name__ == "__main__":
    main()
