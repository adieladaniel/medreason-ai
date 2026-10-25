/*
  Analyze a case: the form, the live progress of one case, and its results.

  The form is built from /api/input-spec, so the labs, units and limits shown here are the ones the server
  enforces. A submitted case is followed through /api/cases/<id>/stream (server-sent events), with a polling
  fallback, and each section of the result is drawn the moment its stage has stored something. Every piece of
  text that came from the server or a paper goes through esc() before it reaches the page.

  Reuses the helpers app.js exposes as window.MedReason, so the page looks like the rest of the dashboard.
*/
(() => {
  'use strict';
  const M = window.MedReason;
  const { esc, pct, card, pageHead, CLASSES, CCOL, MNAME } = M;
  const main = document.getElementById('main');

  const STAGES = [
    ['predicting', 'Prediction', 'Each model reads its input and the results are combined.'],
    ['explaining', 'Explanations', 'A heat map, lab contributions and word contributions.'],
    ['retrieving', 'Evidence lookup', 'Searching the paper library for each finding.'],
    ['writing', 'Writing', 'A small local model summarises each passage.'],
    ['verifying', 'Checking', 'Each sentence is checked against its passage.'],
  ];
  const TERMINAL = ['done', 'partial', 'failed'];
  // below this the reliability analysis (scripts/eval_uncertainty.py) declines to answer; the page says so instead of
  // headlining a weak guess in large type
  const LOW_CONFIDENCE = 0.55;
  // made-up values, not a patient: they only show what the fields expect
  const EXAMPLE = {
    indication: 'Fever, productive cough and shortness of breath for three days. Evaluate for pneumonia.',
    labs: { wbc: 18.5, rbc: 4.1, hemoglobin: 12.0, hematocrit: 36, platelets: 320, neutrophils: 86, lymphocytes: 8,
            monocytes: 5, crp: 120, albumin: 2.9, ldh: 310, sodium: 133, potassium: 4.1, glucose: 128, creatinine: 0.9, urea: 24 },
  };

  /* ------------------------------------------------------------ state */
  let spec = null;
  let token = 0;            // bumped whenever the page is left or another case is opened, so late callbacks do nothing
  let es = null, pollTimer = null, tickTimer = null;
  let current = null;       // {id, createdAt, status}
  const drawn = {};         // last html per section, so an unchanged section is not redrawn (no flicker, no image reload)

  const $ = (sel) => main.querySelector(sel);
  const secs = (s) => (s == null ? '' : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`);
  const safeUrl = (u) => (typeof u === 'string' && /^https?:\/\//i.test(u) ? esc(u) : '');
  const badge = (cls) => `<span class="badge" style="background:${CCOL[cls] || '#64748b'}">${esc(cls)}</span>`;

  function stop() {
    token += 1;
    if (es) { es.close(); es = null; }
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
  }

  /* ------------------------------------------------------------ the page */
  async function analyzePage(caseId) {
    stop();
    for (const k of Object.keys(drawn)) delete drawn[k];
    let html = pageHead('Analyze a case',
      'Give the system a chest X-ray, a clinical indication, blood labs, or any mix of them. It predicts Normal, Pneumonia or Lung Cancer, shows how each input moved the result, and adds cited background from published papers.');
    html += '<div class="callout">This is a demonstration of how the system works, not a medical device, and its prediction is not medical advice. The prediction comes only from the trained models. The cited papers explain it and never change it. Do not enter real patient data: there is no login.</div>';
    try {
      const r = await fetch('/api/input-spec');
      if (!r.ok) throw new Error(r.status);
      spec = await r.json();
    } catch (e) {
      return { html: html + card('', '<p class="muted">The case API is not reachable. Start the server with <code>python -m uvicorn app.main:app --port 8000</code> and reload.</p>'), after() {} };
    }
    html += formCard() + '<div id="result"></div>' +
      card('Recent cases', '<div id="recent"><p class="muted">Loading...</p></div>', { sub: 'Cases are deleted after 30 days.' });
    return {
      html,
      after() {
        M.onLeave(stop);
        wireForm();
        loadRecent();
        if (caseId) openCase(caseId);
      },
    };
  }

  function formCard() {
    const groups = spec.lab_groups.map((g) => {
      const labs = spec.labs.filter((l) => l.group === g).map((l) =>
        `<label class="lab"><span>${esc(l.label)} <em>${esc(l.unit)}</em></span><input type="number" step="any" min="0" inputmode="decimal" data-lab="${esc(l.name)}"></label>`).join('');
      return `<fieldset class="lab-group"><legend>${esc(g)}</legend><div class="lab-grid">${labs}</div></fieldset>`;
    }).join('');
    return `<form id="case-form" class="card form" novalidate>
      <div class="form-cols">
        <div>
          <label class="field"><span>Chest X-ray <em>${spec.image.formats.join(' or ')}, up to ${spec.image.max_mb} MB, optional</em></span>
            <input type="file" id="f-image" accept="image/png,image/jpeg"></label>
          <div id="preview" class="preview" hidden><img id="preview-img" alt="The selected X-ray"><button type="button" class="linkbtn" id="clear-image">Remove</button></div>
          <p class="err" data-err="image"></p>
          <label class="field"><span>Clinical indication <em id="count">0 / ${spec.indication.max_chars}</em></span>
            <textarea id="f-ind" rows="5" maxlength="${spec.indication.max_chars}" placeholder="The reason the X-ray was requested, in a sentence or two. Optional."></textarea></label>
          <p class="err" data-err="indication"></p>
        </div>
        <div>
          <h3 class="form-h">Blood labs <em>All optional. Leave a field empty if it was not measured. Units are the ones shown.</em></h3>
          ${groups}
          <p class="err" data-err="labs"></p>
        </div>
      </div>
      <div class="form-actions">
        <button type="submit" class="btn primary" id="f-go">Analyze</button>
        <button type="button" class="btn" id="f-example">Fill example values</button>
        <button type="button" class="btn ghost" id="f-clear">Clear</button>
        <span class="muted" id="form-note">At least one input is needed. More inputs give a better-supported answer.</span>
      </div>
      <p class="err" data-err="form"></p>
    </form>`;
  }

  /* ------------------------------------------------------------ the form */
  function showErr(field, message) {
    const p = $(`[data-err="${field}"]`) || $('[data-err="form"]');
    p.textContent = message;
    p.classList.add('on');
  }
  function clearErr() { main.querySelectorAll('.err').forEach((p) => { p.textContent = ''; p.classList.remove('on'); }); }

  function wireForm() {
    const form = $('#case-form'), file = $('#f-image'), ta = $('#f-ind');
    const setPreview = () => {
      const f = file.files[0];
      $('#preview').hidden = !f;
      if (f) $('#preview-img').src = URL.createObjectURL(f);
    };
    file.addEventListener('change', () => {
      clearErr();
      const f = file.files[0];
      if (f && f.size > spec.image.max_mb * 1024 * 1024) {
        showErr('image', `That file is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${spec.image.max_mb} MB.`);
        file.value = '';
      } else if (f && !/^image\/(png|jpeg)$/.test(f.type)) {
        showErr('image', 'Only PNG and JPEG images are accepted.');
        file.value = '';
      }
      setPreview();
    });
    $('#clear-image').addEventListener('click', () => { file.value = ''; setPreview(); });
    ta.addEventListener('input', () => { $('#count').textContent = `${ta.value.length} / ${spec.indication.max_chars}`; });
    $('#f-example').addEventListener('click', () => {
      ta.value = EXAMPLE.indication;
      ta.dispatchEvent(new Event('input'));
      main.querySelectorAll('[data-lab]').forEach((i) => { i.value = EXAMPLE.labs[i.dataset.lab] != null ? EXAMPLE.labs[i.dataset.lab] : ''; });
      $('#form-note').textContent = 'These are made-up values, not a patient. Add a chest X-ray of your own if you have one.';
    });
    $('#f-clear').addEventListener('click', () => {
      // not form.reset(): a control whose id or name is "reset" would shadow that method
      HTMLFormElement.prototype.reset.call(form); setPreview(); clearErr();
      $('#count').textContent = `0 / ${spec.indication.max_chars}`;
    });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      clearErr();
      const fd = new FormData();
      if (file.files[0]) fd.append('image', file.files[0]);
      if (ta.value.trim()) fd.append('indication', ta.value);
      const labs = {};
      main.querySelectorAll('[data-lab]').forEach((i) => { if (i.value.trim() !== '') labs[i.dataset.lab] = Number(i.value); });
      if (Object.keys(labs).length) fd.append('labs', JSON.stringify(labs));
      if (![...fd.keys()].length) { showErr('form', 'Give at least one of: a chest X-ray, a clinical indication, or a lab value.'); return; }
      const go = $('#f-go');
      go.disabled = true; go.textContent = 'Sending...';
      try {
        const r = await fetch('/api/cases', { method: 'POST', body: fd });
        const body = await r.json().catch(() => ({}));
        if (r.status === 202) {
          history.replaceState(null, '', `#/analyze/${body.id}`);
          openCase(body.id);
          loadRecent();
        } else if (Array.isArray(body.detail)) {
          body.detail.forEach((d) => showErr(d.field, d.message));
        } else {
          showErr('form', typeof body.detail === 'string' ? body.detail : `The server answered ${r.status}.`);
        }
      } catch (err) {
        showErr('form', 'Could not reach the server. Check that it is still running.');
      } finally {
        go.disabled = false; go.textContent = 'Analyze';
      }
    });
  }

  /* ------------------------------------------------------------ following a case */
  async function openCase(id) {
    stop();
    const mine = token;
    current = { id, createdAt: null, status: 'queued' };
    for (const k of Object.keys(drawn)) delete drawn[k];
    $('#result').innerHTML = `<h2 class="section">Result</h2>
      <div id="r-status"></div><div id="r-pred"></div><div id="r-mods"></div><div id="r-evid"></div><div id="r-audit"></div>`;
    $('#result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const r = await fetch(`/api/cases/${id}`);
      if (mine !== token) return;
      if (r.status === 404) { $('#r-status').innerHTML = alertBox('warn', 'No such case', 'It may have been deleted or have expired.'); return; }
      render(await r.json());
    } catch (e) { /* the stream below will try again */ }
    if (mine !== token || (current && TERMINAL.includes(current.status))) return;
    connect(id, mine);
  }

  function connect(id, mine) {
    if (!window.EventSource) { poll(id, mine); return; }
    es = new EventSource(`/api/cases/${id}/stream`);
    es.addEventListener('case', (e) => { if (mine === token) render(JSON.parse(e.data)); });
    es.addEventListener('done', () => { if (mine === token) { es.close(); es = null; loadRecent(); } });
    es.addEventListener('gone', () => { if (mine === token) { es.close(); es = null; } });
    es.onerror = () => {                       // the browser would reconnect and replay; polling is simpler to reason about
      if (es) { es.close(); es = null; }
      if (mine === token) poll(id, mine);
    };
  }

  async function poll(id, mine) {
    if (mine !== token) return;
    try {
      const r = await fetch(`/api/cases/${id}`);
      if (r.ok) {
        const v = await r.json();
        if (mine !== token) return;
        render(v);
        if (TERMINAL.includes(v.status)) { loadRecent(); return; }
      }
    } catch (e) { /* try again */ }
    pollTimer = setTimeout(() => poll(id, mine), 1500);
  }

  /* ------------------------------------------------------------ drawing a case */
  function put(sel, html) {
    if (drawn[sel] === html) return;
    drawn[sel] = html;
    const el = $(sel);
    if (el) el.innerHTML = html;
  }
  const alertBox = (kind, title, text) => `<div class="alert ${kind}"><b>${esc(title)}</b>${text ? ` ${esc(text)}` : ''}</div>`;

  function render(v) {
    current = { id: v.id, createdAt: Date.parse(v.created_at), status: v.status };
    put('#r-status', statusHtml(v));
    put('#r-pred', v.prediction ? predictionHtml(v) : '');
    put('#r-mods', v.prediction ? modalitiesHtml(v) : '');
    put('#r-evid', v.evidence || TERMINAL.includes(v.status) ? evidenceHtml(v) : '');
    put('#r-audit', auditHtml(v));
    if (!TERMINAL.includes(v.status) && !tickTimer) {
      tickTimer = setInterval(() => { const e = $('#elapsed'); if (e && current) e.textContent = secs((Date.now() - current.createdAt) / 1000); }, 1000);
    }
    if (TERMINAL.includes(v.status) && tickTimer) { clearInterval(tickTimer); tickTimer = null; }
  }

  function stageState(v, key) {
    const ev = v.events.filter((e) => e.stage === key);
    if (ev.some((e) => e.status === 'failed')) return 'failed';
    if (ev.some((e) => e.status === 'finished')) return 'done';
    if (ev.some((e) => e.status === 'started')) return v.status === 'running' ? 'running' : 'stopped';
    return TERMINAL.includes(v.status) ? 'skipped' : 'pending';
  }

  function statusHtml(v) {
    const items = STAGES.map(([key, name, hint]) => {
      const st = stageState(v, key);
      const icon = st === 'running' ? '<span class="spin"></span>' : st === 'done' ? '&#10003;' : st === 'failed' ? '!' : '';
      return `<li class="stage ${st}" title="${esc(hint)}"><span class="st-ico">${icon}</span><span class="st-name">${name}</span><span class="st-sec">${secs(v.timings[key])}</span></li>`;
    }).join('');
    const total = Object.values(v.timings).reduce((a, b) => a + b, 0);
    let line;
    if (v.status === 'queued') line = 'Waiting for the worker to pick the case up.';
    else if (v.status === 'running') line = `Working. <span id="elapsed">${secs((Date.now() - Date.parse(v.created_at)) / 1000)}</span> so far. The prediction appears first; explanations and evidence follow.`;
    else if (v.status === 'done') line = `Finished in ${secs(total)} of work.`;
    else line = '';
    let notice = '';
    if (v.status === 'failed') notice = alertBox('bad', 'The case failed.', v.error || '');
    if (v.status === 'partial') notice = alertBox('warn', 'Part of the analysis did not finish.', `${(v.error || '').replace(/[.\s]+$/, '')}. What is shown below is what was completed.`);
    return `<section class="card"><ol class="stages">${items}</ol><p class="stage-line">${line}</p>${notice}</section>`;
  }

  function probRows(probs) {
    return CLASSES.map((c) => {
      const p = probs[c] || 0;
      return `<div class="prob"><span>${c}</span><span class="bar"><i style="width:${(p * 100).toFixed(0)}%;background:${CCOL[c]}"></i></span><span>${pct(p)}</span></div>`;
    }).join('');
  }

  function predictionHtml(v) {
    const p = v.prediction, b = p.belief, cls = b.predicted_class, c = p.consistency;
    const used = p.modalities_used.map((m) => MNAME[m]).join(', ');
    const cpill = { agree: ['yes', 'The models agree'], partial: ['partial', 'Two of three agree'], conflict: ['no', 'The models disagree'], single_modality: ['todo', 'One input only'] }[c.status] || ['todo', c.status];
    const top = (pr) => CLASSES.reduce((a, c) => ((pr[c] || 0) > (pr[a] || 0) ? c : a), CLASSES[0]);
    const history = b.history.map((h, i) => {
      const own = top(h.probabilities), after = top(h.belief_after);
      return `<tr><td>${i + 1}</td><td>${MNAME[h.modality]}</td><td>${badge(own)} ${pct(h.probabilities[own])}</td><td>${badge(after)} ${pct(h.belief_after[after])}</td></tr>`;
    }).join('');
    const plan = p.missing_info_plan || {};
    const rec = plan.recommendation ? `<div class="callout">${esc(plan.recommendation)}</div>` : '';
    const low = b.confidence < LOW_CONFIDENCE;
    const lowNote = low
      ? `<span class="pill no">Low confidence</span><span class="k-sub">Below ${LOW_CONFIDENCE}, the level at which the reliability analysis declines to answer. Treat this as a weak guess, not a finding.</span>`
      : '';
    return `<section class="card pred${low ? ' weak' : ''}">
        <div class="pred-main"><span class="k-label">Predicted by the trained models</span>
          <span class="pred-class" style="color:${CCOL[cls]}">${esc(cls)}</span>
          <span class="k-sub">${pct(b.confidence)} probability, combining ${esc(used)}</span>${lowNote}</div>
        <div class="pred-probs">${probRows(b.probabilities)}</div>
      </section>
      <section class="card"><h3>Do the inputs agree? <span class="pill ${cpill[0]}">${cpill[1]}</span></h3>
        <p>${esc(c.detail)}</p>
        <p class="sub" style="margin-top:12px">How the answer was built, one input at a time</p>
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Step</th><th>Input</th><th>On its own</th><th>Combined so far</th></tr></thead><tbody>${history}</tbody></table></div>
        ${rec}
      </section>`;
  }

  function modalitiesHtml(v) {
    const p = v.prediction;
    const cols = ['image', 'blood', 'text'].map((m) => {
      const pm = p.modalities[m], ex = v.explanations && v.explanations[m];
      if (!pm) {
        return `<div class="mod absent"><h4>${MNAME[m]}<span class="badge off">not provided</span></h4><p class="summary">Nothing was given for this input, so it took no part in the prediction.</p></div>`;
      }
      let body = '';
      if (m === 'image' && v.inputs.image_url) body += `<img src="${esc(v.inputs.image_url)}" alt="The uploaded X-ray" loading="lazy">`;
      if (ex && ex.figure) {
        body += `<a href="${esc(ex.figure)}" target="_blank" rel="noopener"><img src="${esc(ex.figure)}" alt="${MNAME[m]} explanation" loading="lazy"></a>`;
      } else if (ex && ex.figure_error) {
        body += `<p class="summary">The figure could not be drawn: ${esc(ex.figure_error)}</p>`;
      } else if (!ex && !TERMINAL.includes(v.status)) {
        body += '<p class="waiting"><span class="spin"></span> Computing the explanation...</p>';
      }
      const chips = ex ? ex.attributions.items.slice(0, 5).map((t) => {
        const val = t.value != null && t.value !== '' ? ` ${esc(t.value)}` : '';
        return `<span class="chip ${t.effect > 0 ? 'up' : 'down'}">${esc(t.feature)}${val} ${t.effect > 0 ? '+' : ''}${Number(t.effect).toFixed(2)}</span>`;
      }).join('') : '';
      const warn = ex && ex.matches_prediction === false
        ? `<p class="summary warnline">The explanation run gave different probabilities (largest difference ${esc(ex.max_probability_difference)}). The prediction above is the stored one and was not changed.</p>` : '';
      return `<div class="mod"><h4>${MNAME[m]}${badge(pm.predicted_class)}</h4>${body}
        ${ex ? `<p class="summary">${esc(ex.summary)}</p>` : ''}
        <div class="probs">${probRows(pm.probabilities)}</div>
        ${chips ? `<div class="chips">${chips}</div>` : ''}${warn}</div>`;
    }).join('');
    return `<h2 class="section">What each input contributed</h2>
      <p class="section-sub">Red marks push toward the predicted class, blue push away. The heat map shows where the image model looked, the bars show which labs counted, and the shading shows which words counted.</p>
      <div class="grid3 mods">${cols}</div>`;
  }

  function evidenceHtml(v) {
    const e = v.evidence;
    const head = '<h2 class="section">Evidence from the literature</h2>';
    if (!e) return head + card('', `<p class="muted">${v.status === 'failed' ? 'No evidence was gathered because the case failed.' : 'The evidence stages did not run.'}</p>`);
    const queries = (e.plan && e.plan.queries) || [];
    if (!e.final) {
      const looked = queries.map((q) => `<span class="chip">${esc(q.text)}</span>`).join('');
      const msg = e.written ? 'Checking each sentence against its passage...' : (e.passages ? 'Writing a sentence for each passage found...' : 'Searching...');
      const failed = TERMINAL.includes(v.status);
      return head + card('', `<p class="${failed ? 'muted' : 'waiting'}">${failed ? 'The note was not finished.' : `<span class="spin"></span> ${msg}`}</p>${looked ? `<p class="sub">Looked up</p><div class="chips">${looked}</div>` : ''}`);
    }
    const f = e.final, w = e.written || {};
    const refs = (f.references || []).map((r) => {
      const ps = (e.passages || []).find((x) => x.n === r.n) || {};
      const url = safeUrl(r.url);
      return `<li><b>[${r.n}]</b> ${esc(r.citation || '')}${ps.title ? `, <i>${esc(ps.title)}</i>` : ''}${url ? ` <a href="${url}" target="_blank" rel="noopener">open paper</a>` : ''}
        ${ps.text ? `<details><summary>Show the passage this sentence was written from${ps.heading ? ` (${esc(ps.heading)})` : ''}</summary><p class="passage">${esc(ps.text)}</p></details>` : ''}</li>`;
    }).join('');
    const bg = (f.verified || []).map((s) => `<li>${esc(s.text)} <sup class="cite">[${s.n}]</sup></li>`).join('');
    const notes = [];
    if (e.plan && e.plan.note) notes.push(e.plan.note);
    if ((w.no_evidence_for || []).length) notes.push(`The paper library has no passage for: ${w.no_evidence_for.join('; ')}.`);
    if ((f.dropped || []).length) notes.push(`${f.dropped.length} drafted sentence(s) were removed because they did not match the passage they cited.`);
    if ((w.not_addressed || []).length) notes.push(`${w.not_addressed.length} retrieved passage(s) were judged off topic and not used.`);
    return head + `<section class="card">
        <h3>About this case</h3>
        <ul class="plain">${(w.case_sentences || []).map((s) => `<li>${esc(s)}</li>`).join('')}</ul>
        <h3 style="margin-top:16px">What published papers say</h3>
        <div class="callout">This is background from papers, summarised by a small local model and checked against the passage it came from. It is not proof of anything about this patient, and the checker can miss a sentence that mixes true facts into a false claim. Open the passage to read what the paper actually says.</div>
        ${bg ? `<ul class="plain">${bg}</ul>` : '<p class="muted">No sentence passed the check, so there is no literature note for this case.</p>'}
        ${notes.length ? `<ul class="plain notes">${notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : ''}
        ${refs ? `<h3 style="margin-top:16px">References</h3><ul class="plain refs">${refs}</ul>` : ''}
        <p class="note">Written by ${esc(f.model || w.model || 'a local model')}${f.pass_rate != null ? `; ${pct(f.pass_rate)} of its drafted sentences passed the check` : ''}.</p>
      </section>`;
  }

  function auditHtml(v) {
    const rows = STAGES.filter(([k]) => v.timings[k] != null).map(([k, name]) => `<tr><td>${name}</td><td class="num">${secs(v.timings[k])}</td></tr>`).join('');
    const ev = v.events.map((e) => `<tr><td>${esc(new Date(e.ts).toLocaleTimeString())}</td><td>${esc(e.stage)}</td><td>${esc(e.status)}</td><td class="num">${secs(e.seconds)}</td></tr>`).join('');
    const finished = TERMINAL.includes(v.status);
    return `<h2 class="section">How this was produced</h2><section class="card">
      ${rows ? `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Stage</th><th class="num">Seconds</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<p class="muted">No stage has finished yet.</p>'}
      <details><summary>Every step, with times</summary><div class="tbl-wrap"><table class="tbl"><thead><tr><th>Time</th><th>Stage</th><th>Event</th><th class="num">Seconds</th></tr></thead><tbody>${ev}</tbody></table></div></details>
      <p class="audit-links"><a href="/api/cases/${esc(v.id)}" target="_blank" rel="noopener">Raw case (JSON)</a>
        <a href="/api/cases/${esc(v.id)}/llm-calls" target="_blank" rel="noopener">Every prompt sent to the language model</a>
        ${finished ? '<button type="button" class="linkbtn danger" id="del">Delete this case and its files</button>' : ''}</p></section>`;
  }

  /* delete is wired by delegation because the audit section is redrawn */
  main.addEventListener('click', async (e) => {
    if (!e.target || e.target.id !== 'del' || !current) return;
    if (!confirm('Delete this case and its uploaded files? This cannot be undone.')) return;
    const id = current.id;
    const r = await fetch(`/api/cases/${id}`, { method: 'DELETE' });
    if (r.status === 204 || r.status === 404) {
      stop();
      history.replaceState(null, '', '#/analyze');
      const res = $('#result');
      if (res) res.innerHTML = '';
      loadRecent();
    } else {
      alert(`The server refused: ${r.status}`);
    }
  });

  /* ------------------------------------------------------------ recent cases */
  async function loadRecent() {
    const box = $('#recent');
    if (!box) return;
    try {
      const r = await fetch('/api/cases?limit=8');
      const rows = (await r.json()).cases;
      if (!$('#recent')) return;
      const st = { done: 'done', partial: 'partial', failed: 'no', running: 'partial', queued: 'todo' };
      box.innerHTML = rows.length
        ? `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Submitted</th><th>Status</th><th>Prediction</th><th>Inputs</th><th></th></tr></thead><tbody>${rows.map((c) =>
          `<tr><td>${esc(new Date(c.created_at).toLocaleString())}</td><td><span class="pill ${st[c.status] || 'todo'}">${esc(c.status)}</span></td>
            <td>${c.predicted_class ? `${badge(c.predicted_class)} ${pct(c.confidence)}` : '<span class="muted">none yet</span>'}</td>
            <td>${(c.modalities_used || []).map((m) => MNAME[m]).join(', ') || ''}</td>
            <td><a href="#/analyze/${esc(c.id)}">Open</a></td></tr>`).join('')}</tbody></table></div>`
        : '<p class="muted">No cases yet.</p>';
    } catch (e) { box.innerHTML = '<p class="muted">Could not load recent cases.</p>'; }
  }

  M.routes.analyze = analyzePage;
})();
