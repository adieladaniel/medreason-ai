"""scripts/prepare_public_demo.py: the conversion, the credit line and the output location. The download itself needs the network and is
not tested; what is tested is that nothing about the picture except its channel count changes, and that nothing is written outside data/."""
import importlib.util
import io
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("prepare_public_demo", ROOT / "scripts" / "prepare_public_demo.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def tinted_jpeg(size=(120, 90)):
    """A grayscale ramp with a colour cast, saved as a JPEG: what the CDC file is like."""
    ramp = np.tile(np.linspace(20, 235, size[0]), (size[1], 1))
    rgb = np.stack([ramp, ramp * 0.97, ramp * 0.8], axis=-1).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def test_the_film_is_made_single_channel_without_changing_its_size_or_brightness_order(tmp_path):
    raw = tinted_jpeg()
    film, accepted, reason = prep.prepare(raw, tmp_path)
    out = Image.open(film)
    assert out.mode == "L" and out.size == (120, 90)
    original = np.asarray(Image.open(io.BytesIO(raw)).convert("L")).astype(int)
    assert np.array_equal(np.asarray(out).astype(int), original)                    # the same picture, only on one channel
    assert isinstance(accepted, bool) and (accepted or reason)                        # the guard's verdict and, if refused, why


def test_the_credit_line_names_the_source_the_photographer_and_the_conversion(tmp_path):
    prep.prepare(tinted_jpeg(), tmp_path)
    credit = (tmp_path / "CREDIT.txt").read_text(encoding="utf-8")
    assert "CDC/ Dr. Jack Poland" in credit and "Public domain" in credit and prep.SOURCE_PAGE in credit and "grayscale" in credit


def test_by_default_the_film_goes_under_data_which_git_ignores_and_the_source_is_the_cdc_library():
    assert prep.OUT_DIR == ROOT / "data" / "demo_public"
    assert "data/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert prep.IMAGE_URL.startswith("https://wwwn.cdc.gov/phil") and "17298" in prep.IMAGE_URL
