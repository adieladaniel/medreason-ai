/*
  The frame around every page: the sliding highlight in the floating nav, its icon-only tooltips on narrower screens, the Ctrl+K
  "jump to" palette and the entrance of each page's content. It reads the nav links that index.html holds, so a new page appears
  here without a change.
  Nothing in this file is needed for a page to work; if it fails, navigation and the pages still do.
*/
(() => {
  'use strict';
  const nav = document.getElementById('nav');
  const ind = nav.querySelector('.nav-ind');
  const main = document.getElementById('main');
  const calm = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ------------------------------------------------------------ the highlight under the active link */
  function place(instant) {
    const a = nav.querySelector('a.active');
    if (!a) { ind.style.opacity = '0'; return; }
    if (instant || calm()) ind.classList.add('still');
    ind.style.opacity = '1';
    ind.style.setProperty('--ix', `${a.offsetLeft}px`);
    ind.style.setProperty('--iy', `${a.offsetTop}px`);
    ind.style.setProperty('--iw', `${a.offsetWidth}px`);
    ind.style.setProperty('--ih', `${a.offsetHeight}px`);
    if (nav.scrollWidth > nav.clientWidth + 1) {                 // the phone layout scrolls the nav sideways: keep the active page in view
      nav.scrollTo({ left: Math.max(0, a.offsetLeft - (nav.clientWidth - a.offsetWidth) / 2), behavior: instant || calm() ? 'auto' : 'smooth' });
    }
    if (ind.classList.contains('still')) { void ind.offsetWidth; ind.classList.remove('still'); }
  }
  // the router toggles .active on the links; watch for that (and ignore the highlight's own class changes)
  new MutationObserver((records) => { if (records.some((r) => r.target.tagName === 'A')) place(false); })
    .observe(nav, { attributes: true, attributeFilter: ['class'], subtree: true });
  window.addEventListener('resize', () => place(true));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => place(true));
  place(true);

  /* ------------------------------------------------------------ icon-only width: each icon needs a name on hover */
  const compact = window.matchMedia('(min-width: 801px) and (max-width: 1199px)');   // keep in step with style.css
  function names() {
    nav.querySelectorAll('a[data-page]').forEach((a) => { if (compact.matches) a.title = a.textContent.trim(); else a.removeAttribute('title'); });
    place(true);
  }
  compact.addEventListener('change', names);
  names();

  /* ------------------------------------------------------------ each page's content rises in once */
  new MutationObserver(() => {
    main.classList.remove('enter');
    void main.offsetWidth;                                        // restart the animation
    main.classList.add('enter');
  }).observe(main, { childList: true });

  /* ------------------------------------------------------------ Ctrl+K: jump to a page */
  const plus = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14"/><path d="M12 5v14"/></svg>';
  const items = [...nav.querySelectorAll('a[data-page]')].map((a) => ({
    label: a.textContent.trim(), href: a.getAttribute('href'), icon: a.querySelector('svg').outerHTML,
  }));
  items.splice(1, 0, { label: 'Start a new case', href: '#/analyze', icon: plus });
  let pal = null, list = null, input = null, shown = [], at = 0, before = null;

  function build() {
    pal = document.createElement('div');
    pal.className = 'pal';
    pal.hidden = true;
    pal.innerHTML = `<div class="pal-backdrop"></div>
      <div class="pal-box" role="dialog" aria-modal="true" aria-label="Jump to a page">
        <input type="text" class="pal-input" placeholder="Jump to a page..." autocomplete="off" spellcheck="false" role="combobox" aria-expanded="true" aria-controls="pal-list" aria-label="Page name">
        <ul class="pal-list" id="pal-list" role="listbox"></ul>
        <p class="pal-hint"><kbd>Up</kbd> <kbd>Down</kbd> move <kbd>Enter</kbd> open <kbd>Esc</kbd> close</p>
      </div>`;
    document.body.appendChild(pal);
    list = pal.querySelector('.pal-list');
    input = pal.querySelector('.pal-input');
    pal.querySelector('.pal-backdrop').addEventListener('click', close);
    input.addEventListener('input', () => { at = 0; draw(); });
    list.addEventListener('mousemove', (e) => {
      const li = e.target.closest('li');
      if (li && Number(li.dataset.i) !== at) { at = Number(li.dataset.i); mark(); }
    });
    list.addEventListener('click', (e) => { const li = e.target.closest('li'); if (li) go(shown[Number(li.dataset.i)]); });
    pal.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); close(); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); at = Math.min(shown.length - 1, at + 1); mark(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); at = Math.max(0, at - 1); mark(); }
      else if (e.key === 'Enter') { e.preventDefault(); if (shown[at]) go(shown[at]); }
      else if (e.key === 'Tab') e.preventDefault();               // the palette is one field and a list: focus stays in the field
    });
  }
  function draw() {
    const q = input.value.trim().toLowerCase();
    shown = items.filter((it) => it.label.toLowerCase().includes(q));
    list.innerHTML = shown.length
      ? shown.map((it, i) => `<li role="option" data-i="${i}" aria-selected="${i === at}">${it.icon}<span>${it.label}</span></li>`).join('')
      : '<li class="none" role="presentation">No page matches that.</li>';
    mark();
  }
  function mark() {
    list.querySelectorAll('li[data-i]').forEach((li) => {
      const on = Number(li.dataset.i) === at;
      li.setAttribute('aria-selected', String(on));
      if (on) li.scrollIntoView({ block: 'nearest' });
    });
  }
  function open() {
    if (!pal) build();
    before = document.activeElement;
    input.value = ''; at = 0; draw();
    pal.hidden = false;
    input.focus();
  }
  function close() {
    if (!pal || pal.hidden) return;
    pal.hidden = true;
    if (before && before.focus) before.focus();
  }
  function go(it) {
    close();
    location.hash = it.href;           // from a finished case (#/analyze/<id>) to #/analyze draws an empty form, so "new case" needs nothing more
  }
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (pal && !pal.hidden) close(); else open();
    }
  });
  document.getElementById('jump').addEventListener('click', open);
})();
