"""
What the API accepts, and why each limit is there. Every function here is pure (bytes or text in, a cleaned
value or an InputError out), so the rules can be tested without a server.

The upload is the part that can hurt: a client controls the bytes. So the size is capped while reading, the
format is decided from the file's contents and not its name or declared type, the pixel count is checked
before the image is decoded (a small PNG can unpack to gigabytes), and the whole image is decoded once here,
so a truncated file is refused now and not by the model halfway through a job. Only PNG and JPEG are taken.

What is not checked: whether the picture is a chest X-ray. Any valid PNG or JPEG is accepted, and a photo of a
cat would get a prediction. A guard for that is planned (ROADMAP, Phase 8b) and is not built.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass

from PIL import Image

from app import xray_guard

# The 17 raw lab columns the blood model was trained on, in the same order as explain.blood.RAW_LABS (a test
# keeps the two equal; this copy exists so the API does not have to import the model libraries to check input).
LAB_NAMES = ["wbc", "rbc", "hemoglobin", "hematocrit", "platelets", "neutrophils", "lymphocytes", "monocytes",
             "crp", "esr", "albumin", "ldh", "sodium", "potassium", "glucose", "creatinine", "urea"]
# For the form: a readable name, the unit MIMIC recorded the value in (the blood model was trained on these
# units, so a value typed in another unit gives a wrong prediction without any error), and a group.
# Units were read from the lab data itself, and the training medians match them (hemoglobin 9.7 g/dL).
LAB_INFO = {
    "wbc": ("White blood cells", "K/uL", "Blood count"),
    "rbc": ("Red blood cells", "m/uL", "Blood count"),
    "hemoglobin": ("Hemoglobin", "g/dL", "Blood count"),
    "hematocrit": ("Hematocrit", "%", "Blood count"),
    "platelets": ("Platelets", "K/uL", "Blood count"),
    "neutrophils": ("Neutrophils", "% of white cells", "White cell differential"),
    "lymphocytes": ("Lymphocytes", "% of white cells", "White cell differential"),
    "monocytes": ("Monocytes", "% of white cells", "White cell differential"),
    "crp": ("C-reactive protein", "mg/L", "Often not measured"),
    "esr": ("Sedimentation rate", "mm/hr", "Often not measured"),
    "albumin": ("Albumin", "g/dL", "Often not measured"),
    "ldh": ("Lactate dehydrogenase", "IU/L", "Often not measured"),
    "sodium": ("Sodium", "mEq/L", "Chemistry"),
    "potassium": ("Potassium", "mEq/L", "Chemistry"),
    "glucose": ("Glucose", "mg/dL", "Chemistry"),
    "creatinine": ("Creatinine", "mg/dL", "Chemistry"),
    "urea": ("Urea nitrogen", "mg/dL", "Chemistry"),
}
MAX_LAB_VALUE = 1_000_000.0       # no real lab is near this; it only stops absurd or overflowing input
MIN_IMAGE_SIDE = 64
IMAGE_EXTENSIONS = {"PNG": "png", "JPEG": "jpg"}


class InputError(ValueError):
    """A request the API refuses. `status` is the HTTP code, `field` names the form field at fault."""

    def __init__(self, field: str, message: str, status: int = 422):
        super().__init__(message)
        self.field, self.message, self.status = field, message, status


def input_spec(cfg) -> dict:
    """What the form may send, built from the same constants the checks use, so the page cannot offer a lab
    the server would refuse or a limit the server does not enforce. `cfg` is the Settings object."""
    return {
        "image": {"formats": ["PNG", "JPEG"], "max_mb": cfg.max_upload_mb, "min_side": MIN_IMAGE_SIDE,
                  "max_pixels": cfg.max_image_pixels},
        "indication": {"max_chars": cfg.max_indication_chars},
        "labs": [{"name": n, "label": LAB_INFO[n][0], "unit": LAB_INFO[n][1], "group": LAB_INFO[n][2]} for n in LAB_NAMES],
        "lab_groups": list(dict.fromkeys(LAB_INFO[n][2] for n in LAB_NAMES)),
    }


def clean_indication(raw: str | None, max_chars: int) -> str | None:
    """The indication, trimmed, or None when blank. Over-long text is refused, not cut: silently dropping
    the end of what someone typed would change what the model reads without telling them."""
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    if len(text) > max_chars:
        raise InputError("indication", f"the indication is {len(text)} characters; the limit is {max_chars}")
    return text


def parse_labs(raw: str | None) -> dict | None:
    """Lab values from a JSON object such as {"wbc": 12.5, "albumin": 2.8}, or None when there are none.
    Only the 17 known names are allowed (a typo should be an error, not a silently ignored value), a value
    of null means "not measured", and numbers must be finite and not negative."""
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        raise InputError("labs", "labs must be a JSON object, for example {\"wbc\": 12.5, \"albumin\": 2.8}") from None
    if not isinstance(data, dict):
        raise InputError("labs", "labs must be a JSON object, not a list or a single value")
    unknown = sorted(set(data) - set(LAB_NAMES))
    if unknown:
        raise InputError("labs", f"unknown lab name(s): {', '.join(unknown)}. Known names: {', '.join(LAB_NAMES)}")
    out = {}
    for name, value in data.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError("labs", f"{name} must be a number, not {type(value).__name__}")
        if not math.isfinite(value) or value < 0 or value > MAX_LAB_VALUE:
            raise InputError("labs", f"{name} = {value} is not a plausible lab value (a finite number from 0 to {MAX_LAB_VALUE:g})")
        out[name] = float(value)
    return out or None


@dataclass(frozen=True)
class ImageInfo:
    extension: str
    width: int
    height: int


def check_image(data: bytes, max_pixels: int, guard: bool = False) -> ImageInfo:
    """Accepts a PNG or JPEG that decodes completely and is a sensible size. With `guard`, also refuses what is clearly not a
    radiograph (app/xray_guard.py): a colour photograph, a blank or flat picture, a gradient, a page of text. The guard is a gate
    at the door and does not know whether the film is a chest or frontal."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt, (width, height) = img.format, img.size       # reads the header only
    except Exception:                                      # noqa: BLE001  (PIL raises many types for non-images)
        raise InputError("image", "the file is not an image this server can read (PNG or JPEG only)") from None
    if fmt not in IMAGE_EXTENSIONS:
        raise InputError("image", f"{fmt or 'this'} images are not accepted; send a PNG or a JPEG")
    if width * height > max_pixels:
        raise InputError("image", f"the image is {width} by {height} pixels; the limit is {max_pixels:,} pixels in total")
    if min(width, height) < MIN_IMAGE_SIDE:
        raise InputError("image", f"the image is {width} by {height} pixels; each side must be at least {MIN_IMAGE_SIDE}")
    try:
        img.load()                                          # decode everything now, so a truncated file fails here
    except Exception:                                      # noqa: BLE001
        raise InputError("image", "the image file is damaged or cut short and cannot be decoded") from None
    if guard:
        looks_ok, reason, _ = xray_guard.check(img)
        if not looks_ok:
            raise InputError("image", f"{reason}. Send a chest X-ray as a PNG or JPEG.")
    return ImageInfo(IMAGE_EXTENSIONS[fmt], width, height)
