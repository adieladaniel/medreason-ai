"""The Analyze page in a real browser (Playwright's Chromium), against the real app, API and pipeline code.

Only the models are fake: predictors, explainers, the retriever, the language model and the entailment
checker return canned results with small delays, so each in-between state of a running case can be seen and
the whole file runs in about a minute. Slow-marked because it needs Playwright and its browser:

    pip install playwright && python -m playwright install chromium
    python -m pytest tests/test_analyze_page.py -m slow

Two real bugs were found by running this page for the first time: a button with the id "reset" shadowed
the form's own reset method (so Clear did nothing), and a display rule overrode the hidden attribute (so an
empty X-ray placeholder showed before any file was chosen). Both have a test below.
"""
import io
import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("playwright")
from PIL import Image                                   # noqa: E402
from playwright.sync_api import sync_playwright         # noqa: E402

import app.rag.retrieve as R                            # noqa: E402
from app.agent import scope as scope_check               # noqa: E402
from app.config import Settings                         # noqa: E402
from app.main import create_app                         # noqa: E402
from app.pipelines.runner import JobRunner              # noqa: E402
from app.rag import faithfulness, writer                # noqa: E402

pytestmark = pytest.mark.slow


def probs(n, p, c):
    return {"Normal": n, "Pneumonia": p, "Lung Cancer": c}


P = {"text": probs(0.05, 0.90, 0.05), "blood": probs(0.10, 0.80, 0.10), "image": probs(0.30, 0.60, 0.10)}
ATT = {
    "blood": [{"feature": "WBC", "value": 18.5, "effect": 0.62, "direction": "supports"},
              {"feature": "albumin", "value": 2.9, "effect": 0.31, "direction": "supports"}],
    "image": [{"feature": "Lung Opacity", "value": 0.71, "effect": 0.4, "direction": "supports"}],
    "text": [{"feature": "pneumonia", "value": None, "effect": 0.4, "direction": "supports"}],
}
DELAY = {"blood": 0.3, "image": 0.4, "text": 0.5}
NAME = {"image": "Image", "blood": "Blood", "text": "Text"}
PASSAGES = {
    "pneumonia with elevated white blood cell count": {"chunk_id": 1, "text": "Leukocytosis was common in pneumonia and tracked with severity.", "heading": "Discussion",
                                                       "citation": "Author 1 et al., Journal 2026", "title": "A study of leukocytosis", "url": "https://example.org/paper/1", "rerank_score": 9.0},
    "pneumonia with low albumin": {"chunk_id": 2, "text": "Low albumin was associated with increased mortality in pneumonia.", "heading": "Results",
                                   "citation": "Author 2 et al., Journal 2026", "title": "A study of albumin", "url": "https://example.org/paper/2", "rerank_score": 9.0},
}


ACTIVE = {"P": P}       # a test may swap in other probabilities for the next case


def prediction(m):
    p = ACTIVE["P"][m]
    out = {"modality": m, "predicted_class": max(p, key=p.get), "probabilities": p}
    if m == "image":
        out["_backbone_outputs"] = np.zeros(18)           # what the real image predictor also returns, for the scope check
    return out


def explainer(m):
    def run(_inp):
        time.sleep(DELAY[m])
        return {**prediction(m), "attributions": {"method": "fake", "items": ATT[m]}, "summary": f"{NAME[m]} model predicts Pneumonia.",
                "_cam": np.zeros((2, 2))}
    return run


def plot(_res, path):
    Image.new("RGB", (300, 160), (90, 130, 200)).save(path)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analyze_page")
    mp = pytest.MonkeyPatch()
    state = {"llm_down": False, "markup": False}
    def retrieve(q, k=5):
        hit = PASSAGES.get(q)
        if hit and state["markup"]:       # a paper's own text may contain anything, and so may a model's answer
            hit = {**hit, "text": hit["text"] + ' <script>window.__pwned=true</script><b id="pwn2">bold</b>', "title": '<i id="pwn3">title</i>'}
        return [hit] if hit else []
    mp.setattr(R, "retrieve", retrieve)

    def ask(passage, _topic, _model, _timeout):
        time.sleep(0.3)
        if state["llm_down"]:
            raise ConnectionRefusedError("ollama is not running")
        if state["markup"]:
            return 'Low albumin <img src=x onerror="window.__pwned=true"> <b id="pwn">is</b> associated with mortality in pneumonia.'
        return "Low albumin is associated with increased mortality in pneumonia." if "albumin" in passage["text"].lower() else "Leukocytosis is common in pneumonia."
    mp.setattr(writer, "_ask", ask)
    mp.setattr(faithfulness, "entailment", lambda s, p: 0.9)
    state["scope"] = None                                 # a test sets this to what the (fake) scope check returns for the next case
    mp.setattr(scope_check, "assess", lambda *a, **k: state["scope"])

    demo = tmp / "demo"
    (demo / "fake-case").mkdir(parents=True)
    Image.effect_noise((300, 300), 40).convert("L").save(demo / "fake-case" / "image.jpg", "JPEG")
    (demo / "manifest.json").write_text(json.dumps({"cases": [{
        "id": "fake-case", "title": "A fake validation case", "note": "Why this one: it is made up for the test.",
        "indication": "Cough for two days. Evaluate for pneumonia.", "labs": {"wbc": 14.2, "albumin": 3.1},
        "study_id": 123456, "expected": {"true_label": "Normal"}}]}), encoding="utf-8")
    url = "sqlite:///" + (tmp / "page.db").as_posix()
    cfg = Settings(database_url=url, upload_dir=tmp / "uploads", demo_dir=demo)
    deps = {"predictors": {m: (lambda _i, m=m: prediction(m)) for m in P}, "explainers": {m: explainer(m) for m in P},
            "plotters": {m: plot for m in P}}
    app = create_app(cfg, JobRunner(url, deps=deps, figure_dir=cfg.upload_dir), build_dashboard=False)
    app.state.poll_interval = 0.2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    buf = io.BytesIO()
    Image.effect_noise((400, 400), 40).convert("L").save(buf, "PNG")
    (tmp / "film.png").write_bytes(buf.getvalue())
    yield SimpleNamespace(base=f"http://127.0.0.1:{port}", state=state, film=tmp / "film.png")
    server.should_exit = True
    mp.undo()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, site):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    pg.problems = []
    pg.on("pageerror", lambda e: pg.problems.append(f"page error: {e}"))
    # a refused request (422, 413) is a normal answer to bad input and the browser logs it; anything else is a problem
    pg.on("console", lambda m: pg.problems.append(f"console: {m.text}") if m.type == "error" and "status of 4" not in m.text else None)
    pg.goto(f"{site.base}/#/analyze")
    pg.wait_for_selector("#case-form")
    yield pg
    assert pg.problems == []                       # no script error on any path the test took
    ctx.close()


EXAMPLE_TEXT = "Fever, productive cough and shortness of breath for three days. Evaluate for pneumonia."
EXAMPLE_LABS = {"wbc": 18.5, "rbc": 4.1, "hemoglobin": 12.0, "hematocrit": 36, "platelets": 320, "neutrophils": 86, "lymphocytes": 8,
                "monocytes": 5, "crp": 120, "albumin": 2.9, "ldh": 310, "sodium": 133, "potassium": 4.1, "glucose": 128, "creatinine": 0.9,
                "urea": 24}


def fill_example(pg):
    """Type a pneumonia-like indication and a full set of labs, as a person would. (The page used to have a button for this; it was
    replaced by a menu of complete example cases, because made-up values next to someone's own X-ray gave Pneumonia every time.)"""
    pg.fill("#f-ind", EXAMPLE_TEXT)
    for name, value in EXAMPLE_LABS.items():
        pg.fill(f'[data-lab="{name}"]', str(value))


def finish(pg, timeout=60000):
    pg.wait_for_function("document.querySelector('#r-audit #del')", timeout=timeout)


# ------------------------------------------------------------------ the form
def test_the_form_is_built_from_the_servers_spec(page):
    assert page.locator("[data-lab]").count() == 17
    assert [t.lower() for t in page.locator(".lab-group legend").all_inner_texts()] == [
        "blood count", "white cell differential", "often not measured", "chemistry"]
    form = page.inner_text("#case-form")
    assert "g/dL" in form and "K/uL" in form and "10 MB" in form
    assert page.locator("#nav a.active").inner_text() == "Analyze a case"


def test_no_empty_x_ray_placeholder_before_a_file_is_chosen(page):
    """The hidden attribute used to lose to a display rule, leaving a broken image on the page."""
    assert not page.locator("#preview").is_visible()


def test_choosing_a_file_shows_a_preview_and_remove_hides_it(page, site):
    page.set_input_files("#f-image", str(site.film))
    assert page.locator("#preview").is_visible()
    page.click("#clear-image")
    assert not page.locator("#preview").is_visible()


def test_the_clear_button_empties_the_form(page):
    """A control with the id "reset" used to shadow form.reset(), so Clear threw and did nothing."""
    fill_example(page)
    assert page.input_value('[data-lab="albumin"]') == "2.9" and "pneumonia" in page.input_value("#f-ind")
    page.click("#f-clear")
    assert page.input_value('[data-lab="albumin"]') == "" and page.input_value("#f-ind") == ""
    assert page.inner_text("#count").startswith("0 /")


def test_an_empty_form_is_refused_without_contacting_the_server(page):
    sent = []
    page.on("request", lambda r: sent.append(r.url) if r.method == "POST" else None)
    page.click("#f-go")
    page.wait_for_function("document.querySelector('[data-err=form]').textContent.length > 0")
    assert "at least one" in page.inner_text('[data-err="form"]').lower() and sent == []


def test_the_servers_reasons_appear_next_to_the_field(page):
    page.fill('[data-lab="wbc"]', "-5")
    page.click("#f-go")
    page.wait_for_function("document.querySelector('[data-err=labs]').textContent.length > 0")
    assert "plausible" in page.inner_text('[data-err="labs"]')
    page.click("#f-clear")
    page.set_input_files("#f-image", {"name": "notes.png", "mimeType": "image/png", "buffer": b"this is not an image"})
    page.fill("#f-ind", "cough")
    page.click("#f-go")
    page.wait_for_function("document.querySelector('[data-err=image]').textContent.length > 0")
    assert "not an image" in page.inner_text('[data-err="image"]')


# ------------------------------------------------------------------ a case, live
def test_a_case_is_followed_live_and_every_section_arrives(page, site):
    fill_example(page)
    page.set_input_files("#f-image", str(site.film))
    page.click("#f-go")
    page.wait_for_selector("#r-pred .pred-class")
    assert page.locator("#r-pred .pred-class").inner_text() == "Pneumonia"
    assert page.locator("#r-status .stage.running, #r-status .stage.pending").count() > 0      # the answer came before the end
    case_id = page.url.split("/")[-1]
    assert len(case_id) == 32
    finish(page)
    assert page.locator("#r-status .stage.done").count() == 5
    assert all(page.eval_on_selector_all("#r-mods .mod img", "els => els.map(e => e.complete && e.naturalWidth > 0)"))
    assert page.locator("#r-mods .mod img").count() >= 4                       # the X-ray and three explanation figures
    evidence = page.inner_text("#r-evid").lower()
    assert "not proof" in evidence and page.locator("#r-evid .cite").count() >= 1
    assert page.locator('#r-evid a[href^="https://example.org"]').count() >= 1
    assert "Leukocytosis" in page.inner_html("#r-evid")                        # the passage a sentence came from can be opened
    assert "so far" not in page.inner_text("#r-status")


def test_a_past_case_can_be_reopened_from_the_list_and_deleted(page, site):
    fill_example(page)
    page.click("#f-go")
    finish(page)
    page.goto(f"{site.base}/#/analyze")
    page.reload()
    page.wait_for_selector("#recent a")
    page.locator("#recent a").first.click()
    finish(page, 20000)
    assert page.locator("#r-pred .pred-class").inner_text() == "Pneumonia"
    n = page.evaluate("fetch('/api/cases?limit=100').then(r => r.json()).then(j => j.cases.length)")
    page.once("dialog", lambda d: d.accept())
    page.click("#del")
    page.wait_for_function("!document.querySelector('#r-audit')")
    assert page.url.endswith("#/analyze")
    assert page.evaluate("fetch('/api/cases?limit=100').then(r => r.json()).then(j => j.cases.length)") == n - 1


def test_a_case_with_the_language_model_down_says_which_stage_failed(page, site):
    site.state["llm_down"] = True
    try:
        fill_example(page)
        page.click("#f-go")
        finish(page)
    finally:
        site.state["llm_down"] = False
    status = page.inner_text("#r-status")
    assert "did not finish" in status and "writing failed" in status and "ollama is not running." in status
    assert page.locator("#r-pred .pred-class").inner_text() == "Pneumonia"      # the prediction is still shown


def test_other_dashboard_pages_still_render_and_the_dashboard_is_not_needed_for_analyze(page):
    page.click('#nav a[data-page="models"]')
    page.wait_for_function("document.querySelector('h1') && document.querySelector('h1').textContent !== 'Analyze a case'")
    page.click('#nav a[data-page="analyze"]')
    page.wait_for_selector("#case-form")


# ------------------------------------------------------------------ small screens
def test_nothing_scrolls_sideways_at_phone_width(page, site):
    page.set_viewport_size({"width": 390, "height": 800})
    page.reload()
    page.wait_for_selector("#case-form")
    assert page.evaluate("document.documentElement.scrollWidth") <= page.evaluate("document.documentElement.clientWidth") + 2
    fill_example(page)
    page.click("#f-go")
    finish(page)
    assert page.evaluate("document.documentElement.scrollWidth") <= page.evaluate("document.documentElement.clientWidth") + 2


def test_markup_in_paper_text_or_model_output_is_shown_as_text_and_never_runs(page, site):
    site.state["markup"] = True
    try:
        fill_example(page)
        page.click("#f-go")
        finish(page)
    finally:
        site.state["markup"] = False
    assert page.evaluate("window.__pwned") is None                         # no script or handler ran
    for tag_id in ("pwn", "pwn2", "pwn3"):
        assert page.locator(f"#{tag_id}").count() == 0, tag_id              # the tags were not turned into elements
    assert "<b id=" in page.inner_text("#r-evid") or "&lt;" in page.inner_html("#r-evid")      # the text itself is still there, visibly


def run_with(page, probabilities, click_example=True, image=None):
    """Runs one case whose three fake models answer with `probabilities`, then restores the confident default."""
    ACTIVE["P"] = probabilities
    try:
        if click_example:
            fill_example(page)
        if image:
            page.set_input_files("#f-image", str(image))
        page.click("#f-go")
        finish(page)
    finally:
        ACTIVE["P"] = P


def test_a_low_confidence_answer_is_headlined_as_uncertain_with_the_best_guess_below(page):
    run_with(page, {m: probs(0.38, 0.40, 0.22) for m in P})          # the lung cancer case the real models get wrong
    text = page.inner_text("#r-pred")
    assert page.locator("#r-pred .pred-class").inner_text() == "Uncertain"
    assert "Best guess by probability" in text and page.locator("#r-pred .pill", has_text="Low confidence").count() == 1
    assert page.locator("#r-pred .pred.weak").count() == 1 and "weak guess" in text
    assert page.locator("#r-pred .set-badges .badge").count() == 3 and "None of the classes can be ruled out" in text
    assert "follow-up" not in text.lower()


def test_a_confident_answer_keeps_its_class_headline_and_a_one_class_set(page):
    run_with(page, P)
    assert page.locator("#r-pred .pred-class").inner_text() == "Pneumonia"
    assert page.locator("#r-pred .pill", has_text="Low confidence").count() == 0
    assert page.locator("#r-pred .set-badges .badge").all_inner_texts() == ["Pneumonia"]
    assert "Only this class remains possible" in page.inner_text("#r-pred") and "validation cases a set like this held" in page.inner_text("#r-pred")
    assert page.locator("#r-pred .alert").count() == 0


def test_a_cancer_signal_under_another_headline_shows_the_alert_with_its_numbers(page):
    from app.agent.decision import decide
    from app.agent.reasoning import reason
    signal = {"text": probs(0.10, 0.50, 0.40), "blood": probs(0.20, 0.45, 0.35), "image": probs(0.30, 0.40, 0.30)}
    belief = reason(signal)["belief"]
    d = decide(belief["probabilities"], list(signal))
    assert belief["predicted_class"] == "Pneumonia" and d["cancer_alert"]["flag"], "the scenario must really contain a cancer signal"
    run_with(page, signal)
    banner = page.inner_text("#r-pred .alert")
    assert "Lung Cancer signal" in banner and "not a diagnosis" in banner and "alerts per 100 cases" in banner
    assert page.locator("#r-pred .set-badges .badge", has_text="Lung Cancer").count() == 1
    assert page.locator("#r-pred .pred-class").inner_text() in ("Pneumonia", "Uncertain")        # the headline is still a model prediction


def test_an_alert_that_is_no_better_than_chance_is_not_shown_and_the_page_says_why(page, site):
    image_only = {m: (probs(0.2, 0.4, 0.4) if m == "image" else P[m]) for m in P}
    run_with(page, image_only, click_example=False, image=site.film)
    assert page.locator("#r-pred .alert").count() == 0
    assert "No cancer alert is offered for this combination of inputs" in page.inner_text("#r-pred")


def test_a_set_that_leaves_out_the_most_probable_class_says_why(page):
    """Each class has its own bar, so a near-even case can give a set holding only the rare class. That is correct for the
    method but looks wrong without an explanation."""
    from app.agent.decision import decide
    even = {m: probs(0.34, 0.33, 0.33) for m in P}
    assert decide({"Normal": 0.34, "Pneumonia": 0.33, "Lung Cancer": 0.33}, list(P))["prediction_set"] == ["Lung Cancer"]
    run_with(page, even)
    assert page.locator("#r-pred .set-badges .badge").all_inner_texts() == ["Lung Cancer"]
    assert "most probable class is not in the set" in page.inner_text("#r-pred")
