"""The frame around the pages (floating nav, jump palette, motion, font) in a real browser. Uses the site, browser and page fixtures of
test_analyze_page.py. Slow-marked: needs Playwright and its browser."""
import pytest

pytest.importorskip("playwright")
from tests.test_analyze_page import browser, fill_example, finish, page, site  # noqa: E402,F401  (fixtures)

pytestmark = pytest.mark.slow

BOX = "e => { const r = e.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; }"


def settle(pg, ms=500):
    pg.wait_for_timeout(ms)                      # the highlight takes 240 ms to arrive


def test_the_highlight_sits_under_the_active_link_and_follows_navigation(page):
    settle(page)
    active = page.locator("#nav a.active")
    assert active.inner_text() == "Analyze a case"
    assert page.eval_on_selector(".nav-ind", BOX) == active.evaluate(BOX)
    page.click('#nav a[data-page="models"]')
    page.wait_for_function("document.querySelector('#nav a.active').dataset.page === 'models'")
    settle(page)
    assert page.eval_on_selector(".nav-ind", BOX) == page.locator('#nav a[data-page="models"]').evaluate(BOX)


def test_the_vendored_typeface_is_actually_served_and_used(page):
    page.wait_for_function("document.fonts.status === 'loaded'")
    assert page.evaluate("document.fonts.check('16px Inter')")
    assert page.evaluate("[...document.fonts].some(f => f.family.replace(/\"/g, '') === 'Inter' && f.status === 'loaded')")


def test_ctrl_k_opens_the_palette_filters_it_and_enter_goes_there(page):
    assert not page.locator(".pal").count() or not page.locator(".pal").is_visible()
    page.keyboard.press("Control+k")
    page.wait_for_selector(".pal-input", state="visible")
    assert page.evaluate("document.activeElement.className") == "pal-input"                      # typing works at once
    assert page.locator(".pal-list li[data-i]").count() == 10                                    # nine pages and "start a new case"
    page.keyboard.type("reli")
    assert page.locator(".pal-list li[data-i]").all_inner_texts() == ["Reliability"]
    page.keyboard.press("Enter")
    page.wait_for_function("document.querySelector('#nav a.active').dataset.page === 'reliability'")
    assert page.url.endswith("#/reliability") and not page.locator(".pal").is_visible()


def test_the_palette_moves_with_the_arrow_keys_closes_on_escape_and_returns_focus(page):
    page.focus("#jump")
    page.keyboard.press("Control+k")
    page.wait_for_selector(".pal-input", state="visible")
    page.keyboard.press("ArrowDown")
    assert page.locator('.pal-list li[aria-selected="true"]').inner_text() == "Start a new case"
    page.keyboard.type("zzz")
    assert page.inner_text(".pal-list") == "No page matches that."
    page.keyboard.press("Enter")                                                                  # nothing to open: stays put
    assert page.locator(".pal").is_visible() and page.url.endswith("#/analyze")
    page.keyboard.press("Escape")
    assert not page.locator(".pal").is_visible()
    assert page.evaluate("document.activeElement.id") == "jump"


def test_start_a_new_case_from_the_palette_resets_a_finished_case(page):
    fill_example(page)
    page.click("#f-go")
    finish(page)
    assert not page.locator("#case-form").is_visible()
    page.keyboard.press("Control+k")
    page.keyboard.type("new case")
    page.keyboard.press("Enter")
    page.wait_for_selector("#case-form", state="visible")
    assert page.input_value("#f-ind") == "" and page.locator("#r-pred").count() == 0


def test_the_nav_is_a_translucent_pill_centred_at_the_top_that_stays_while_the_page_scrolls(page):
    page.goto(page.url.split("#")[0] + "#/fusion")
    page.wait_for_selector("#main h1")
    settle(page, 700)                                                                             # it slides in once on load
    x, y, w, h = page.eval_on_selector(".dock", BOX)
    assert abs((x + w / 2) - 1280 / 2) <= 2, (x, w)                                               # centred in the window, not in the content
    assert y == 14 and h < 70
    position, blur, colour, round_ends = page.eval_on_selector(
        ".dock", "e => { const c = getComputedStyle(e); return [c.position, c.backdropFilter, c.backgroundColor, parseFloat(c.borderRadius) >= e.offsetHeight / 2]; }")
    assert position == "fixed" and "blur" in blur and round_ends
    assert colour.startswith("rgba(") and float(colour.rstrip(")").split(",")[-1]) < 1                # see-through, so the page shows faintly behind it
    page.evaluate("window.scrollTo(0, 600)")
    page.wait_for_timeout(200)
    assert page.eval_on_selector(".dock", BOX) == [x, y, w, h]                                    # floats over the content
    assert page.evaluate("window.scrollY") > 0


def test_there_is_no_sidebar_and_the_page_uses_the_whole_width(page):
    assert page.locator(".side").count() == 0 and page.locator("#collapse").count() == 0
    left, top, width, _ = page.eval_on_selector("#main", BOX)
    assert left < 40 and width > 1200, (left, width)                                              # the old 244 pixel column is gone
    assert top == 0 and page.eval_on_selector("#main h1", BOX)[1] > 70                            # but the heading clears the pill
    assert "validation results" in page.inner_text(".foot")                                       # the note that lived in the sidebar
    assert page.evaluate("document.documentElement.scrollWidth") <= 1281


def test_on_a_narrower_window_the_pill_keeps_its_icons_and_each_has_a_name(page):
    page.set_viewport_size({"width": 1000, "height": 800})
    page.wait_for_timeout(300)
    link = page.locator('#nav a[data-page="models"]')
    assert link.get_attribute("title") == "Models"                                                # an icon alone needs a name
    assert link.evaluate(BOX)[2] < 60                                                             # the word takes no room
    assert page.evaluate("document.documentElement.scrollWidth") <= 1001
    settle(page)
    assert page.eval_on_selector(".nav-ind", BOX) == page.locator("#nav a.active").evaluate(BOX)  # the highlight fits the narrow link
    page.set_viewport_size({"width": 1280, "height": 800})
    page.wait_for_timeout(300)
    assert link.get_attribute("title") is None and link.evaluate(BOX)[2] > 80                     # and the words come back


def test_a_page_blocks_rise_in_and_a_person_who_asks_for_less_motion_gets_none(page):
    page.click('#nav a[data-page="models"]')
    page.wait_for_selector("#main.enter")
    assert page.evaluate("getComputedStyle(document.querySelector('#main > *')).animationName") == "rise"
    page.emulate_media(reduced_motion="reduce")
    page.click('#nav a[data-page="fusion"]')
    page.wait_for_function("document.querySelector('#nav a.active').dataset.page === 'fusion'")
    assert page.evaluate("getComputedStyle(document.querySelector('#main > *')).animationName") == "none"
    assert page.evaluate("getComputedStyle(document.querySelector('.nav-ind')).transitionProperty") == "none"


def test_on_a_phone_the_nav_is_one_scrolling_row_that_keeps_the_active_page_in_view(page):
    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(page.url.split("#")[0] + "#/roadmap")
    page.reload()
    page.wait_for_selector("#main h1")
    settle(page, 700)
    tops = {page.locator(f'#nav a[data-page="{p}"]').evaluate("e => Math.round(e.getBoundingClientRect().top)")
            for p in ("analyze", "overview", "roadmap")}
    assert len(tops) == 1                                                                         # one row
    box = page.locator("#nav a.active").evaluate(BOX)
    assert box[0] >= 0 and box[0] + box[2] <= 390, box                                            # the active link is on screen
    assert not page.locator("#jump").is_visible()
    assert page.eval_on_selector(".dock", "e => getComputedStyle(e).position") == "absolute"       # it scrolls away rather than costing every screen 70 pixels
    assert page.evaluate("document.documentElement.scrollWidth") <= 392
