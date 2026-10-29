"""
The non-X-ray guard (app/xray_guard.py) and its place at the door of the API.

The real films are git-ignored, so the normal suite tests the machinery on synthetic film-like pictures (fit a model on a few hundred,
check that new ones pass and that obvious non-films do not), the committed fitted parameters against pictures that need no film to be
obviously wrong, and the wiring through the upload endpoint. The measurement on real validation films is a slow test.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import xray_guard as g
from app.config import Settings
from app.main import create_app
from app.pipelines.runner import JobRunner
from scripts import build_xray_guard as builder

ROOT = Path(__file__).resolve().parent.parent


def film_like(rng, size=(400, 480)) -> Image.Image:
    """A crude radiograph: dark surround, two bright lung-shaped regions, a bright central column, soft noise."""
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 25.0)
    for cx in (w * 0.3, w * 0.7):
        img += 140 * np.exp(-(((xx - cx) / (w * 0.14)) ** 2 + ((yy - h * 0.45) / (h * 0.3)) ** 2))
    img += 90 * np.exp(-(((xx - w * 0.5) / (w * 0.06)) ** 2 + ((yy - h * 0.5) / (h * 0.4)) ** 2))
    img = img * rng.uniform(0.8, 1.2) + rng.normal(0, 6, (h, w)) + rng.uniform(-15, 15)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "L")


@pytest.fixture(scope="module")
def synthetic_params():
    rng = np.random.default_rng(0)
    fit = np.stack([g.thumbnail(film_like(rng)) for _ in range(300)])
    cal = [film_like(rng) for _ in range(150)]
    return g.build_params(fit, np.stack([g.thumbnail(i) for i in cal]), np.array([g.extreme_fraction(i) for i in cal]))


# ------------------------------------------------------------------ the cues
def test_colourfulness_is_zero_for_gray_in_any_mode_and_large_for_colour():
    gray = Image.fromarray(np.random.default_rng(1).integers(0, 255, (80, 80), dtype=np.uint8), "L")
    assert g.colourfulness(gray) == 0.0
    assert g.colourfulness(gray.convert("RGB")) == 0.0                  # a gray picture saved with three equal channels
    red = Image.new("RGB", (80, 80), (230, 20, 20))
    assert g.colourfulness(red) > 100


def test_structure_is_near_zero_for_a_ramp_and_large_for_a_pattern():
    ramp = np.tile(np.linspace(0, 255, g.THUMB), (g.THUMB, 1)).ravel()
    assert g.structure(ramp) < 1.0
    lumpy = np.random.default_rng(2).normal(128, 50, g.THUMB * g.THUMB)
    assert g.structure(lumpy) > 30


def test_extreme_fraction_counts_pure_black_and_white_pixels():
    a = np.full((100, 100), 128, dtype=np.uint8)
    a[:, :50] = 255
    assert abs(g.extreme_fraction(Image.fromarray(a, "L")) - 0.5) < 0.05
    assert g.extreme_fraction(Image.fromarray(np.full((100, 100), 128, dtype=np.uint8), "L")) == 0.0


def test_standardise_removes_brightness_and_contrast():
    t = np.random.default_rng(3).normal(100, 20, 400)
    assert np.allclose(g.standardise(t), g.standardise(t * 1.7 + 30))


# ------------------------------------------------------------------ the machinery on synthetic films
def test_new_film_like_pictures_pass_and_the_thresholds_were_set_from_held_out_pictures(synthetic_params):
    rng = np.random.default_rng(10)
    passed = [g.check(film_like(rng), synthetic_params)[0] for _ in range(100)]
    assert np.mean(passed) >= 0.97
    for key in ("max_distance", "min_spread", "min_structure", "max_extreme_fraction", "max_colourfulness", "residual_mean"):
        assert synthetic_params[key] > 0


def test_the_threshold_comes_from_held_out_films_so_a_small_fitting_set_does_not_make_it_too_tight():
    # With only 40 films to fit 20 components, the fitted films sit unrealistically close to the model. A threshold taken from
    # them would refuse many new films; one taken from other films does not.
    rng = np.random.default_rng(21)
    fit = np.stack([g.thumbnail(film_like(rng)) for _ in range(40)])
    cal = [film_like(rng) for _ in range(300)]
    p = g.build_params(fit, np.stack([g.thumbnail(i) for i in cal]), np.array([g.extreme_fraction(i) for i in cal]))
    in_sample = np.percentile(g.distances(fit, p), 99.5)
    assert p["max_distance"] > 2 * in_sample                        # the held-out distances are much larger than the in-sample ones
    new = [g.check(film_like(rng), p)[0] for _ in range(150)]
    assert np.mean(new) >= 0.97


def test_every_kind_of_obvious_non_film_is_rejected_with_a_reason(synthetic_params):
    rng = np.random.default_rng(11)
    for kind, images in builder.negatives(rng, 6).items():
        if "not required" in kind:
            continue
        for im in images:
            ok, reason, _ = g.check(im, synthetic_params)
            assert not ok and reason, kind


@pytest.mark.parametrize("make,fragment", [
    (lambda: Image.new("RGB", (300, 300), (200, 30, 30)), "colour photograph"),
    (lambda: Image.new("L", (300, 300), 120), "no large-scale structure"),
    (lambda: Image.fromarray(np.tile(np.linspace(0, 255, 300), (300, 1)).astype(np.uint8), "L"), "smooth gradient"),
    (lambda: Image.fromarray(np.random.default_rng(5).integers(0, 255, (300, 300), dtype=np.uint8), "L"), "no large-scale structure"),
    (lambda: Image.fromarray((40 + 150 * np.exp(-(((np.mgrid[0:300, 0:300][1] - 60) / 40.0) ** 2 + ((np.mgrid[0:300, 0:300][0] - 240) / 40.0) ** 2))
                             ).astype(np.uint8), "L"), "layout"),                    # one bright blob in a corner: structured, unlike a film
])
def test_each_cue_gives_its_own_reason(synthetic_params, make, fragment):
    ok, reason, info = g.check(make(), synthetic_params)
    assert not ok and fragment in reason and info


def test_a_page_that_is_mostly_white_is_refused_for_that_reason(synthetic_params):
    page = np.full((400, 400), 255, dtype=np.uint8)
    page[100:300, 100:300] = np.random.default_rng(6).integers(0, 255, (200, 200), dtype=np.uint8)
    ok, reason, _ = g.check(Image.fromarray(page, "L"), synthetic_params)
    assert not ok and "pure black or pure white" in reason


def test_without_fitted_parameters_nothing_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "PARAMS_PATH", tmp_path / "missing.json")
    g.load_params.cache_clear()
    try:
        ok, reason, info = g.check(Image.new("RGB", (100, 100), (255, 0, 0)))
    finally:
        g.load_params.cache_clear()
    assert ok and reason is None and info == {"guard": "not fitted"}


# ------------------------------------------------------------------ the committed fitted guard
def test_the_committed_parameters_exist_hold_no_image_and_reject_pictures_that_are_plainly_wrong():
    p = json.loads((ROOT / "app" / "data" / "xray_guard.json").read_text(encoding="utf-8"))
    assert {"mean", "components", "variance", "max_distance", "min_structure", "max_extreme_fraction", "max_colourfulness"} <= set(p)
    assert len(p["mean"]) == g.THUMB ** 2 and (ROOT / "app" / "data" / "xray_guard.json").stat().st_size < 400_000
    rng = np.random.default_rng(12)
    for kind, images in builder.negatives(rng, 5).items():
        if "not required" in kind:
            continue
        assert not any(g.check(im)[0] for im in images), kind


def test_the_guard_needs_no_model_and_no_network():
    out = subprocess.run([sys.executable, "-c", "import sys, app.xray_guard; sys.exit(1 if 'torch' in sys.modules else 0)"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-300:]


# ------------------------------------------------------------------ at the door of the API
def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    def make(guard: bool):
        url = f"sqlite:///{(tmp_path / 'g.db').as_posix()}"
        cfg = Settings(database_url=url, upload_dir=tmp_path / "up", xray_guard=guard)
        runner = JobRunner(url, stages=[], deps={"no_figures": True}, figure_dir=cfg.upload_dir)
        c = TestClient(create_app(cfg, runner, build_dashboard=False))
        c.__enter__()
        made.append(c)
        return c
    made = []
    yield make
    for c in made:
        c.__exit__(None, None, None)


def test_a_photograph_is_refused_at_the_door_when_the_guard_is_on_and_accepted_when_it_is_off(client):
    photo = png(Image.new("RGB", (200, 200), (30, 160, 60)))
    on = client(True).post("/api/cases", files={"image": ("cat.png", photo, "image/png")})
    assert on.status_code == 422
    detail = on.json()["detail"][0]
    assert detail["field"] == "image" and "colour photograph" in detail["message"] and "chest X-ray" in detail["message"]
    assert str(ROOT) not in on.text                                       # no server paths in the answer
    off = client(False).post("/api/cases", files={"image": ("cat.png", photo, "image/png")})
    assert off.status_code == 202


def test_a_refused_picture_stores_no_case_and_no_file(client, tmp_path):
    c = client(True)
    c.post("/api/cases", files={"image": ("n.png", png(Image.fromarray(np.random.default_rng(8).integers(0, 255, (200, 200), dtype=np.uint8), "L")),
                                         "image/png")})
    assert c.get("/api/cases").json()["cases"] == []
    assert not list((tmp_path / "up").glob("*/*"))


def test_a_picture_the_guard_accepts_goes_on_to_be_stored(client, monkeypatch):
    monkeypatch.setattr(g, "check", lambda img, params=None: (True, None, {}))
    r = client(True).post("/api/cases", files={"image": ("film.png", png(Image.new("L", (200, 200), 90)), "image/png")})
    assert r.status_code == 202


def test_the_guard_is_on_by_default_in_production_and_off_in_the_suite():
    assert Settings.model_fields["xray_guard"].default is True
    assert Settings().xray_guard is False                                # conftest sets MEDREASON_XRAY_GUARD=false for the suite


@pytest.mark.slow
def test_on_real_validation_films_at_most_one_percent_are_wrongly_refused():
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    sample = val.sample(400, random_state=0).image_path
    refused = sum(not g.check(Image.open(p))[0] for p in sample)
    assert refused / len(sample) <= 0.01
