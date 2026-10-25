"""What the API accepts, without a server. Images are made in memory with Pillow."""
import io
import json

import pytest
from PIL import Image

from app.api import validation as v

BIG = 50_000_000


def image_bytes(fmt="PNG", size=(128, 128)):
    buf = io.BytesIO()
    Image.new("L", size, 90).save(buf, fmt)
    return buf.getvalue()


# ------------------------------------------------------------------ labs
def test_lab_names_match_the_blood_model():
    from explain.blood import RAW_LABS
    assert v.LAB_NAMES == RAW_LABS


def test_labs_keep_only_the_values_given():
    assert v.parse_labs(json.dumps({"wbc": 12.5, "albumin": 2, "crp": None})) == {"wbc": 12.5, "albumin": 2.0}
    assert v.parse_labs(None) is None and v.parse_labs("  ") is None and v.parse_labs("{}") is None
    assert v.parse_labs(json.dumps({"wbc": None})) is None


@pytest.mark.parametrize("raw,field_fragment", [
    ("not json", "JSON object"),
    ("[1, 2]", "not a list"),
    ('{"wbcc": 5}', "unknown lab name"),
    ('{"wbc": "12"}', "must be a number"),
    ('{"wbc": true}', "must be a number"),
    ('{"wbc": -1}', "plausible"),
    ('{"wbc": 1e9}', "plausible"),
    ('{"wbc": NaN}', "plausible"),
    ('{"wbc": Infinity}', "plausible"),
])
def test_bad_labs_are_refused_with_the_reason(raw, field_fragment):
    with pytest.raises(v.InputError) as e:
        v.parse_labs(raw)
    assert e.value.field == "labs" and field_fragment in e.value.message and e.value.status == 422


def test_the_unknown_lab_message_lists_the_names_that_are_allowed():
    with pytest.raises(v.InputError) as e:
        v.parse_labs('{"hb": 12}')
    assert "hemoglobin" in e.value.message and "hb" in e.value.message


# ------------------------------------------------------------------ indication
def test_indication_is_trimmed_and_blank_means_absent():
    assert v.clean_indication("  fever and cough \n", 100) == "fever and cough"
    assert v.clean_indication("   ", 100) is None and v.clean_indication(None, 100) is None


def test_a_too_long_indication_is_refused_not_cut():
    with pytest.raises(v.InputError) as e:
        v.clean_indication("x" * 101, 100)
    assert e.value.field == "indication" and "101" in e.value.message
    assert v.clean_indication("x" * 100, 100) == "x" * 100


# ------------------------------------------------------------------ images
def test_png_and_jpeg_are_accepted_and_named_by_their_real_format():
    assert v.check_image(image_bytes("PNG"), BIG) == v.ImageInfo("png", 128, 128)
    assert v.check_image(image_bytes("JPEG", (200, 100)), BIG) == v.ImageInfo("jpg", 200, 100)


def test_the_file_name_and_declared_type_play_no_part():
    """A JPEG is a JPEG whatever the client calls it: the decision is made from the bytes alone."""
    assert v.check_image(image_bytes("JPEG"), BIG).extension == "jpg"


@pytest.mark.parametrize("data,fragment", [
    (b"this is a text file, not a picture", "not an image"),
    (b"", "not an image"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "not an image"),
])
def test_files_that_are_not_images_are_refused(data, fragment):
    with pytest.raises(v.InputError) as e:
        v.check_image(data, BIG)
    assert e.value.field == "image" and fragment in e.value.message


def test_other_image_formats_are_refused():
    with pytest.raises(v.InputError) as e:
        v.check_image(image_bytes("GIF"), BIG)
    assert "GIF" in e.value.message and "PNG or a JPEG" in e.value.message


def test_the_pixel_limit_is_checked_before_decoding():
    with pytest.raises(v.InputError) as e:
        v.check_image(image_bytes("PNG", (200, 200)), max_pixels=10_000)
    assert "10,000 pixels" in e.value.message and "200 by 200" in e.value.message


def test_a_tiny_image_is_refused():
    with pytest.raises(v.InputError) as e:
        v.check_image(image_bytes("PNG", (32, 500)), BIG)
    assert "at least 64" in e.value.message


def test_a_truncated_file_is_refused_now_not_by_the_model_later():
    whole = image_bytes("PNG", (256, 256))
    with pytest.raises(v.InputError) as e:
        v.check_image(whole[: len(whole) // 2], BIG)
    assert "damaged" in e.value.message or "not an image" in e.value.message
