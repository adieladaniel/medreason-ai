"""The Analyze page's example-case menu in a real browser. Uses the site, browser and page fixtures of test_analyze_page.py, whose fake
demo folder holds one curated case ("fake-case"). Slow-marked: needs Playwright and its browser."""
import pytest

pytest.importorskip("playwright")
from tests.test_analyze_page import browser, fill_example, page, site  # noqa: E402,F401  (fixtures)

pytestmark = pytest.mark.slow


def test_the_example_menu_offers_curated_cases_and_the_not_an_xray_pictures(page):
    options = page.locator("#f-example option").all_inner_texts()
    assert options[0].startswith("Choose an example case")
    assert "A fake validation case" in options and "Not an X-ray: a colour picture" in options and "Not an X-ray: random noise" in options
    groups = page.locator("#f-example optgroup").evaluate_all("els => els.map(e => e.label)")
    assert groups == ["Validation cases", "Not an X-ray (to see the refusal)"]
    assert page.locator("#f-example").input_value() == ""


def test_choosing_an_example_fills_the_whole_form_including_the_xray_and_shows_why_it_was_chosen(page):
    page.select_option("#f-example", "fake-case")
    page.wait_for_function("document.querySelector('#preview') && !document.querySelector('#preview').hidden")
    assert page.input_value("#f-ind") == "Cough for two days. Evaluate for pneumonia."
    assert page.input_value('[data-lab="wbc"]') == "14.2" and page.input_value('[data-lab="albumin"]') == "3.1"
    assert page.input_value('[data-lab="crp"]') == ""                                  # labs not on record stay empty
    assert page.evaluate("document.querySelector('#f-image').files[0].name") == "fake-case.jpg"
    assert page.evaluate("document.querySelector('#f-image').files[0].type") == "image/jpeg"
    assert "it is made up for the test" in page.inner_text("#example-note")
    assert "123456" not in page.content() and "true_label" not in page.content()      # the answer and the study id stay on the server


def test_choosing_an_example_replaces_what_was_there_and_clear_removes_it(page):
    fill_example(page)
    page.select_option("#f-example", "fake-case")
    page.wait_for_function("document.querySelector('#f-ind').value.startsWith('Cough')")
    assert page.input_value('[data-lab="crp"]') == ""                                  # the earlier typed lab is gone, not mixed in
    page.click("#f-clear")
    assert page.input_value("#f-ind") == "" and page.inner_text("#example-note") == ""
    assert page.evaluate("document.querySelector('#f-image').files.length") == 0
    assert page.input_value("#f-example") == ""


def test_a_built_in_example_loads_only_a_picture(page):
    page.select_option("#f-example", "not-an-xray-photo")
    page.wait_for_function("!document.querySelector('#preview').hidden")
    assert page.input_value("#f-ind") == "" and page.evaluate("document.querySelector('#f-image').files[0].type") == "image/png"
    assert "refuse" in page.inner_text("#example-note")


def test_an_example_that_cannot_be_loaded_says_so_instead_of_leaving_the_form_half_filled(page):
    page.route("**/api/examples/fake-case", lambda route: route.abort())
    page.select_option("#f-example", "fake-case")
    page.wait_for_function("document.querySelector('#example-note').textContent.includes('Could not load')")
    assert page.input_value("#f-ind") == ""
    assert page.problems == ["console: Failed to load resource: net::ERR_FAILED"]     # the aborted request, which this test caused
    page.problems.clear()


# ------------------------------------------------------------------ the scope check on the page
from tests.test_analyze_page import P, run_with  # noqa: E402

SCOPE = {"out_of_scope": True, "score": 0.71, "threshold": 0.42, "flag_rate_in_scope": 0.1, "catch_rate_other": 0.48}


def test_a_flagged_case_is_headlined_as_possibly_outside_what_the_system_can_assess_and_still_shows_the_best_guess(page, site):
    site.state["scope"] = SCOPE
    try:
        run_with(page, P, image=site.film)                  # the check needs a film
    finally:
        site.state["scope"] = None
    head = page.inner_text("#r-pred .pred-main")
    assert "Possibly outside what this system can assess" in head
    assert "Best guess among the three classes it knows: Pneumonia" in head
    assert "0.71" in head and "0.42" in head and "about 10%" in head and "about 48%" in head
    assert "does not say what the condition is" in head and "only knows Normal, Pneumonia and Lung Cancer" in head
    assert "Pneumonia" in page.inner_text("#r-pred .pred-probs")                     # the class probabilities are still all shown
    assert page.locator("#r-pred .card.pred.weak, #r-pred section.pred.weak").count() == 1


def test_a_case_the_check_does_not_flag_is_headlined_as_before(page, site):
    site.state["scope"] = {**SCOPE, "out_of_scope": False, "score": 0.12}
    try:
        run_with(page, P, image=site.film)                  # the check needs a film
    finally:
        site.state["scope"] = None
    head = page.inner_text("#r-pred .pred-main")
    assert "Possibly outside" not in head and head.strip().startswith("Predicted by the trained models") and "Pneumonia" in head


def test_a_case_with_no_scope_result_is_headlined_as_before(page):
    run_with(page, P, image=None)
    assert "Possibly outside" not in page.inner_text("#r-pred")


def test_without_a_film_the_check_is_not_run_even_if_a_detector_would_flag(page, site):
    site.state["scope"] = SCOPE                                  # would flag, but the case has no X-ray, so the check never gets asked
    try:
        run_with(page, P, image=None)
    finally:
        site.state["scope"] = None
    assert "Possibly outside" not in page.inner_text("#r-pred")
