(() => {
  'use strict';

  /* ------------------------------------------------------------ constants */
  const CLASSES = ['Normal', 'Pneumonia', 'Lung Cancer'];
  const CCOL = { Normal: '#2f9e77', Pneumonia: '#3b82f6', 'Lung Cancer': '#e0523d' };
  const MCOL = { image: '#6366f1', blood: '#f59e0b', text: '#14b8a6', fusion: '#0f172a' };
  const MNAME = { image: 'Image', blood: 'Blood', text: 'Text', fusion: 'Fusion' };
  const STATE = { done: 'Done', partial: 'Core done', todo: 'Not started' };
  const TIERCOL = {
    'Tier 1 core': '#3b82f6', 'Tier 1 differential': '#8b5cf6',
    'Tier 2': '#e0523d', 'Tier 3': '#14b8a6',
  };

  const main = document.getElementById('main');
  let D = null;
  let charts = [];

  Chart.defaults.font.family = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#475569';
  Chart.defaults.borderColor = '#e5e8ef';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;

  /* draws a value at the end of each bar (no plugin dependency) */
  const valueLabels = {
    id: 'valueLabels',
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.format) return;
      const ctx = chart.ctx;
      chart.data.datasets.forEach((ds, i) => {
        const meta = chart.getDatasetMeta(i);
        if (meta.hidden) return;
        meta.data.forEach((bar, j) => {
          const v = ds.data[j];
          if (v == null) return;
          ctx.save();
          ctx.fillStyle = '#334155';
          ctx.font = '600 11px system-ui, sans-serif';
          ctx.textBaseline = 'middle';
          if (chart.options.indexAxis === 'y') {
            ctx.textAlign = 'left';
            ctx.fillText(opts.format(v), bar.x + 6, bar.y);
          } else {
            ctx.textAlign = 'center';
            ctx.fillText(opts.format(v), bar.x, bar.y - 8);
          }
          ctx.restore();
        });
      });
    },
  };

  /* ------------------------------------------------------------ helpers */
  const f2 = (x) => (x == null ? 'n/a' : Number(x).toFixed(2));
  const num = (n) => (n == null ? 'n/a' : Number(n).toLocaleString('en-US'));
  const pct = (x, d = 0) => (x == null ? 'n/a' : (x * 100).toFixed(d) + '%');
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function mk(id, cfg) {
    const el = document.getElementById(id);
    if (!el) return null;
    cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false }, cfg.options || {});
    cfg.plugins = (cfg.plugins || []).concat([valueLabels]);
    const c = new Chart(el, cfg);
    charts.push(c);
    return c;
  }

  const pageHead = (t, s) => `<header class="page-head"><h1>${t}</h1><p>${s}</p></header>`;
  const sectionHead = (t, s) => `<h2 class="section">${t}</h2>${s ? `<p class="section-sub">${s}</p>` : ''}`;
  const chartBox = (id, h = 300) => `<div class="chart" style="height:${h}px"><canvas id="${id}"></canvas></div>`;
  const card = (title, body, o = {}) =>
    `<section class="card">${title ? `<h3>${title}</h3>` : ''}${o.sub ? `<p class="sub">${o.sub}</p>` : ''}${body}${o.note ? `<p class="note">${o.note}</p>` : ''}</section>`;
  const kpi = (label, value, sub, cls = '') =>
    `<div class="kpi ${cls}"><span class="k-label">${label}</span><span class="k-val">${value}</span><span class="k-sub">${sub}</span></div>`;
  const missing = (what) =>
    card('', `<p class="muted">${what} is not available in this checkout. Run the experiments, then rebuild with <code>python -m app.dashboard_data</code>.</p>`);
  const pill = (state) => `<span class="pill ${state}">${STATE[state]}</span>`;

  const yF1 = { min: 0, max: 1, ticks: { stepSize: 0.2 }, title: { display: true, text: 'F1 score' } };

  function table(head, rows, opts = {}) {
    const th = head.map((h, i) => `<th class="${opts.numCols && opts.numCols.includes(i) ? 'num' : ''}">${h}</th>`).join('');
    const tr = rows.map((r, ri) =>
      `<tr class="${opts.bestRow === ri ? 'best' : ''}">` +
      r.map((c, i) => `<td class="${opts.numCols && opts.numCols.includes(i) ? 'num' : ''}">${c}</td>`).join('') +
      '</tr>').join('');
    return `<div class="tbl-wrap"><table class="tbl"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
  }

  const classBar = (id, datasets, extra = {}) =>
    mk(id, {
      type: 'bar',
      data: { labels: extra.labels || CLASSES, datasets },
      options: Object.assign({
        scales: { y: yF1, x: { grid: { display: false } } },
        plugins: { valueLabels: { format: (v) => Number(v).toFixed(2) } },
      }, extra.options || {}),
    });

  /* ------------------------------------------------------------ overview */
  function overview() {
    const h = D.headline || {};
    const f = D.fusion;
    const phases = D.phases || [];
    const done = phases.filter((p) => p.state === 'done').length;
    const partial = phases.filter((p) => p.state === 'partial').length;

    let html = pageHead('Project progress',
      'An agentic multi-modal diagnostic assistant for chest X-rays. Three models (image, blood labs, clinical text) are combined into one prediction: Normal, Pneumonia, or Lung Cancer.');

    html += '<div class="grid4">' +
      kpi('Image model', f2(h.image), 'Validation macro F1, MIMIC-CXR DenseNet') +
      kpi('Blood model', f2(h.blood), 'Validation macro F1, gradient-boosted trees') +
      kpi('Text model', f2(h.text), 'Validation macro F1, Bio_ClinicalBERT') +
      kpi('Fusion', f2(h.fusion), h.best_single != null ? `+${f2(h.fusion - h.best_single)} over the best single model` : 'Combined prediction', 'accent') +
      '</div>';

    // findings, built from the data so the numbers cannot drift
    const finds = [];
    if (h.fusion != null) {
      finds.push(`Combining the three models reaches ${f2(h.fusion)} macro F1, ${f2(h.fusion - h.best_single)} above the best single model (${f2(h.best_single)}).`);
    }
    const cm = D.predictions && D.predictions.curves.models;
    if (cm) {
      const a = (m) => f2(cm[m].roc['Lung Cancer'].auc);
      let s = `The rare Lung Cancer class separates the models. ROC AUC: image ${a('image')}, blood ${a('blood')}, text ${a('text')}, fusion ${a('fusion')}.`;
      if (cm.image.roc['Lung Cancer'].auc < 0.6) s += ' The image model is close to chance there, so the text and lab models carry that class.';
      finds.push(s);
    }
    const t = D.models && D.models.text;
    if (t) {
      finds.push(`A text model that reads the full radiology report scores ${f2(t.full_report.logreg.macro_f1)}, but the labels were partly built from those reports. The reported text score uses only the clinical indication (${f2(t.indication.logreg.macro_f1)}).`);
    }
    const ag = D.predictions && D.predictions.agreement.levels;
    if (ag && ag.length >= 2) {
      const all = ag[0], last = ag[ag.length - 1];
      finds.push(`When all three models agree (${pct(all.share)} of cases) fused accuracy is ${pct(all.fused_accuracy)}. When all three disagree it falls to ${pct(last.fused_accuracy)}. That gap is the signal the planned consistency checker will use.`);
    }
    if (finds.length) html += card('What the results say', `<ul class="findings">${finds.map((x) => `<li>${x}</li>`).join('')}</ul>`);

    html += '<div class="grid2">' +
      card('Macro F1 by stage', chartBox('c-macro', 280), { sub: 'Validation set, higher is better.' }) +
      card('F1 by class', chartBox('c-class', 280), {
        sub: 'Image here is the version used as a fusion input (0.48 macro F1). The best standalone run scored 0.50.',
      }) + '</div>';

    html += sectionHead('Phases', `${done} of ${phases.length} complete, ${partial} more with the core done.`);
    html += '<div class="grid3">' + phases.map((p) =>
      `<div class="phase ${p.state}"><div class="p-top"><span class="p-num">Phase ${p.num}</span>${pill(p.state)}</div><h4>${esc(p.title)}</h4><p>${esc(p.blurb)}</p></div>`
    ).join('') + '</div>';

    const after = () => {
      mk('c-macro', {
        type: 'bar',
        data: {
          labels: ['Image', 'Blood', 'Text', 'Fusion'],
          datasets: [{ data: [h.image, h.blood, h.text, h.fusion], backgroundColor: [MCOL.image, MCOL.blood, MCOL.text, MCOL.fusion], borderRadius: 6 }],
        },
        options: {
          plugins: { legend: { display: false }, valueLabels: { format: (v) => Number(v).toFixed(2) } },
          scales: { y: { min: 0, max: 0.8, title: { display: true, text: 'Macro F1' } }, x: { grid: { display: false } } },
        },
      });
      if (f) {
        const sets = ['image', 'blood', 'text'].map((m) => ({
          label: MNAME[m], backgroundColor: MCOL[m], borderRadius: 4,
          data: CLASSES.map((c) => f.single[m].f1_per_class[c]),
        }));
        sets.push({ label: 'Fusion', backgroundColor: MCOL.fusion, borderRadius: 4, data: CLASSES.map((c) => f.methods[0].f1[c]) });
        classBar('c-class', sets, { options: { plugins: { valueLabels: { format: () => '' } } } });
      }
    };
    return { html, after };
  }

  /* ------------------------------------------------------------ data */
  function dataPage() {
    const d = D.data;
    if (!d) return { html: pageHead('Data', '') + missing('Data statistics'), after() {} };
    const fu = d.funnel;
    let html = pageHead('Data',
      'Chest X-ray studies from MIMIC-CXR linked to hospital admissions in MIMIC-IV, labeled from diagnosis codes and radiology report text, then split by patient.');

    html += '<div class="grid4">' +
      kpi('CXR studies', num(fu.studies), `${num(fu.subjects)} patients`) +
      kpi('Linked to an admission', num(fu.linked_studies), `${pct(fu.linked_studies / fu.studies, 1)} of studies`) +
      kpi('Final dataset', num(fu.labeled), `${num(Object.values(d.splits).reduce((a, s) => a + s.subjects, 0))} patients, 3 classes`, 'accent') +
      kpi('Lab-linked admissions', num(fu.admissions), `${num(fu.ambiguous)} ambiguous matches resolved`) +
      '</div>';

    html += '<div class="grid2">' +
      card('From studies to training samples', chartBox('c-funnel', 230), {
        sub: 'A study links to an admission when its timestamp falls inside the admission window.',
        note: d.excluded != null
          ? `${num(d.excluded)} linked studies were dropped because they matched none of the three labels (other findings such as effusion or edema).`
          : '',
      }) +
      card('Class balance by split', chartBox('c-splits', 230), {
        sub: 'Split by patient, so no patient appears in more than one split.',
        note: 'Lung Cancer is about 7 percent of every split, which is why macro F1 is the main metric.',
      }) + '</div>';

    const rows = Object.entries(d.splits).map(([s, v]) => [
      s[0].toUpperCase() + s.slice(1), num(v.n), num(v.subjects),
      ...CLASSES.map((c) => `${num(v.counts[c])} (${pct(v.counts[c] / v.n)})`),
    ]);
    html += card('Splits', table(['Split', 'Samples', 'Patients', ...CLASSES], rows, { numCols: [1, 2, 3, 4, 5] }));

    html += sectionHead('Blood lab coverage', 'The share of linked admissions that have each lab. The chosen 17 labs come in tiers; the sparse ones are treated as optional inputs.');
    html += card('', chartBox('c-cov', 430), {
      note: 'CRP (7.5 percent) and ESR (3.7 percent) are too sparse to require, and the blood model confirms it: they contribute almost nothing.',
    });

    if (d.provenance) {
      const pn = d.provenance.Pneumonia, ca = d.provenance['Lung Cancer'], nm = d.provenance.Normal;
      const sum = (o) => o.icd_only + o.report_only + o.both;
      html += sectionHead('Where the labels came from', 'A label comes from a diagnosis code, a mention in the radiology report, or both. This matters for the text model, because a model that reads the report could partly read its own label.');
      html += card('', chartBox('c-prov', 230), {
        note: `${pct((pn.report_only + pn.both) / sum(pn))} of Pneumonia labels and ${pct((ca.report_only + ca.both) / sum(ca))} of Lung Cancer labels have report evidence, and every Normal label requires the report to read as clean (${num(nm.report_clean)} studies). Text is therefore scored on the clinical indication only, which is never used to label.`,
      });
    }

    html += card('Label rules',
      `<p><b>Pneumonia:</b> a pneumonia diagnosis code or a positive pneumonia mention in the report (checked first, so it wins ties).</p>
       <p><b>Lung Cancer:</b> a lung cancer diagnosis code or a positive mention.</p>
       <p><b>Normal:</b> the report shows no findings and the admission has no pneumonia, cancer, or tuberculosis code.</p>
       <p>Studies with other findings are excluded. Mentions come from a negation-aware spaCy labeler, so "no pneumonia" does not count as pneumonia. Tuberculosis and COVID-19 were dropped as classes: COVID cannot appear because the imaging predates the pandemic, and tuberculosis had only 10 cases.</p>`);

    const after = () => {
      mk('c-funnel', {
        type: 'bar',
        data: {
          labels: ['CXR studies', 'Linked to an admission', 'In the final dataset'],
          datasets: [{ data: [fu.studies, fu.linked_studies, fu.labeled], backgroundColor: ['#c7d2fe', '#818cf8', '#4f46e5'], borderRadius: 6 }],
        },
        options: {
          indexAxis: 'y',
          layout: { padding: { right: 56 } },
          plugins: { legend: { display: false }, valueLabels: { format: (v) => num(v) } },
          scales: { x: { grid: { display: false }, ticks: { display: false } }, y: { grid: { display: false } } },
        },
      });
      const sp = Object.keys(d.splits);
      mk('c-splits', {
        type: 'bar',
        data: {
          labels: sp.map((s) => s[0].toUpperCase() + s.slice(1)),
          datasets: CLASSES.map((c) => ({ label: c, backgroundColor: CCOL[c], data: sp.map((s) => d.splits[s].counts[c]) })),
        },
        options: { scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, title: { display: true, text: 'Samples' } } } },
      });
      const cov = d.blood_coverage.slice().sort((a, b) => b.pct - a.pct);
      const tiers = Object.keys(TIERCOL);
      mk('c-cov', {
        type: 'bar',
        data: {
          labels: cov.map((c) => c.test),
          datasets: tiers.map((t) => ({
            label: t, backgroundColor: TIERCOL[t],
            data: cov.map((c) => (c.tier === t ? c.pct : null)),
          })),
        },
        options: {
          indexAxis: 'y',
          scales: {
            x: { stacked: true, min: 0, max: 100, title: { display: true, text: 'Admissions with the lab (%)' } },
            y: { stacked: true, grid: { display: false } },
          },
        },
      });
      if (d.provenance) {
        const P = d.provenance;
        mk('c-prov', {
          type: 'bar',
          data: {
            labels: ['Pneumonia', 'Lung Cancer', 'Normal'],
            datasets: [
              { label: 'Diagnosis code only', backgroundColor: '#94a3b8', data: [P.Pneumonia.icd_only, P['Lung Cancer'].icd_only, 0] },
              { label: 'Report mention only', backgroundColor: '#818cf8', data: [P.Pneumonia.report_only, P['Lung Cancer'].report_only, 0] },
              { label: 'Both', backgroundColor: '#4f46e5', data: [P.Pneumonia.both, P['Lung Cancer'].both, 0] },
              { label: 'Report read as clean', backgroundColor: '#2f9e77', data: [0, 0, P.Normal.report_clean] },
            ],
          },
          options: {
            indexAxis: 'y',
            scales: { x: { stacked: true, title: { display: true, text: 'Studies' } }, y: { stacked: true, grid: { display: false } } },
          },
        });
      }
    };
    return { html, after };
  }

  /* ------------------------------------------------------------ models */
  function modelsPage() {
    const M = D.models;
    if (!M) return { html: pageHead('Models', '') + missing('Model results'), after() {} };
    let html = pageHead('Models', 'Three unimodal baselines, each scored on the validation set. Every model uses a frozen, domain-appropriate encoder with a small trained head.');

    /* image */
    const img = M.image;
    html += sectionHead('Image: chest X-ray', 'Three iterations. The lever that worked was the choice of pretrained features, not more fine-tuning.');
    const vrows = img.versions.map((v) => [`<b>${v.label}</b>`, esc(v.desc), f2(v.macro_f1), f2(v.f1.Normal), f2(v.f1.Pneumonia), f2(v.f1['Lung Cancer'])]);
    html += card('', table(['Run', 'Setup', 'Macro F1', 'Normal', 'Pneumonia', 'Lung Cancer'], vrows, { numCols: [2, 3, 4, 5], bestRow: img.versions.length - 1 }), {
      note: 'v1 and v2 are recorded in the roadmap (no run logs were kept). v3 comes from the Kaggle GPU run log.',
    });
    html += '<div class="grid2">' +
      card('F1 by class and version', chartBox('c-imgver', 280), { sub: 'Lung Cancer stays low in every image run.' }) +
      card('v3 training run', img.training_curve ? chartBox('c-imgcurve', 280) : '<p class="muted">Training log not available.</p>', {
        sub: img.best_epoch ? `Kaggle GPU, 15 epochs. The best validation epoch was ${img.best_epoch}, and that checkpoint was kept.` : '',
        note: 'Validation macro F1 is noisy from epoch to epoch because the Lung Cancer class has only 117 validation samples.',
      }) + '</div>';

    /* blood */
    const b = M.blood;
    html += sectionHead('Blood: 17 lab values', 'Gradient-boosted trees handle missing labs natively, so no rows were dropped for missing values.');
    if (b) {
      const brow = b.models.map((m) => [m.label, f2(m.macro_f1), f2(m.balanced_accuracy), f2(m.f1.Normal), f2(m.f1.Pneumonia), f2(m.f1['Lung Cancer'])]);
      html += '<div class="grid2">' +
        card('Models compared', table(['Model', 'Macro F1', 'Balanced acc.', 'Normal', 'Pneumonia', 'Cancer'], brow, { numCols: [1, 2, 3, 4, 5], bestRow: brow.length - 1 }), {
          note: `Restricting to the ${num(b.core_lab_rows_n)} validation rows with at least one core lab gives ${f2(b.core_lab_rows_f1)}, about the same. Missing labs are not what limits the blood model; the labs simply carry limited signal for this split.`,
        }) +
        card('Which labs matter', chartBox('c-imp', 300), { sub: 'Drop in validation macro F1 when the lab is shuffled (permutation importance).' }) +
        '</div>';
    } else html += missing('Blood results');

    /* text */
    const t = M.text;
    html += sectionHead('Text: the clinical indication', `Frozen ${esc(t ? t.encoder || 'Bio_ClinicalBERT' : 'Bio_ClinicalBERT')} embeddings with a logistic-regression head. Two input settings, because the labels were partly built from the report text.`);
    if (t) {
      const trow = [
        ['Indication only (used)', f2(t.indication.logreg.macro_f1), f2(t.indication.logreg.f1_per_class.Normal), f2(t.indication.logreg.f1_per_class.Pneumonia), f2(t.indication.logreg.f1_per_class['Lung Cancer'])],
        ['Full report (leaky)', f2(t.full_report.logreg.macro_f1), f2(t.full_report.logreg.f1_per_class.Normal), f2(t.full_report.logreg.f1_per_class.Pneumonia), f2(t.full_report.logreg.f1_per_class['Lung Cancer'])],
      ];
      html += '<div class="grid2">' +
        card('Two settings, logistic-regression head', table(['Input', 'Macro F1', 'Normal', 'Pneumonia', 'Cancer'], trow, { numCols: [1, 2, 3, 4], bestRow: 0 }), {
          note: 'The full-report number is higher only because the model can see the words the labels were derived from. It is shown for comparison and is not used anywhere downstream.',
        }) +
        card('Leakage check', chartBox('c-leak', 250), {
          sub: 'Macro F1 split by how each label was derived.',
          note: 'On labels the report cannot influence (diagnosis code only), reading the whole report barely beats reading only the indication. So the full-report lead comes from leakage, and the indication-only score is close to the real ceiling.',
        }) + '</div>';
    } else html += missing('Text results');

    const after = () => {
      const shades = ['#c7d2fe', '#818cf8', '#4338ca'];
      const cls3 = [...CLASSES, 'Macro F1'];
      mk('c-imgver', {
        type: 'bar',
        data: {
          labels: cls3,
          datasets: img.versions.map((v, i) => ({
            label: v.label, backgroundColor: shades[i], borderRadius: 4,
            data: [v.f1.Normal, v.f1.Pneumonia, v.f1['Lung Cancer'], v.macro_f1],
          })),
        },
        options: { scales: { y: yF1, x: { grid: { display: false } } }, plugins: { valueLabels: { format: () => '' } } },
      });
      if (img.training_curve) {
        const cv = img.training_curve;
        mk('c-imgcurve', {
          type: 'line',
          data: {
            labels: cv.map((r) => r.epoch),
            datasets: [
              { label: 'Validation macro F1', data: cv.map((r) => r.val_macro_f1), borderColor: MCOL.image, backgroundColor: MCOL.image, yAxisID: 'y', tension: 0.25, pointRadius: 3 },
              { label: 'Training loss', data: cv.map((r) => r.train_loss), borderColor: '#94a3b8', backgroundColor: '#94a3b8', yAxisID: 'y1', borderDash: [5, 4], tension: 0.25, pointRadius: 0 },
            ],
          },
          options: {
            scales: {
              x: { title: { display: true, text: 'Epoch' }, grid: { display: false } },
              y: { min: 0.4, max: 0.55, title: { display: true, text: 'Macro F1' } },
              y1: { position: 'right', min: 0.95, max: 1.06, grid: { display: false }, title: { display: true, text: 'Loss' } },
            },
          },
        });
      }
      if (b) {
        const top = b.importance.slice(0, 12);
        mk('c-imp', {
          type: 'bar',
          data: { labels: top.map((r) => r.feature), datasets: [{ data: top.map((r) => r.mean), backgroundColor: MCOL.blood, borderRadius: 4 }] },
          options: {
            indexAxis: 'y',
            plugins: { legend: { display: false }, valueLabels: { format: (v) => Number(v).toFixed(3) } },
            layout: { padding: { right: 36 } },
            scales: { x: { title: { display: true, text: 'Macro F1 drop' } }, y: { grid: { display: false } } },
          },
        });
      }
      if (t) {
        mk('c-leak', {
          type: 'bar',
          data: {
            labels: [`Labels backed by report text (n=${t.indication.by_provenance.report.n})`, `Diagnosis-code-only labels (n=${t.indication.by_provenance.icd_only.n})`],
            datasets: [
              { label: 'Indication only', backgroundColor: MCOL.text, borderRadius: 4, data: [t.indication.by_provenance.report.macro_f1, t.indication.by_provenance.icd_only.macro_f1] },
              { label: 'Full report', backgroundColor: '#cbd5e1', borderRadius: 4, data: [t.full_report.by_provenance.report.macro_f1, t.full_report.by_provenance.icd_only.macro_f1] },
            ],
          },
          options: {
            scales: { y: { min: 0, max: 1, title: { display: true, text: 'Macro F1' } }, x: { grid: { display: false } } },
            plugins: { valueLabels: { format: (v) => Number(v).toFixed(2) } },
          },
        });
      }
    };
    return { html, after };
  }

  /* ------------------------------------------------------------ fusion */
  let curveClass = 'Lung Cancer';
  let confMethod = null;

  function confusionHTML(method) {
    const m = D.fusion.methods.find((x) => x.key === method) || D.fusion.methods[0];
    const rows = m.confusion;
    let h = '<table class="heat"><thead><tr><th></th><th></th>' +
      `<th colspan="3">Predicted</th></tr><tr><th></th><th></th>${CLASSES.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
    rows.forEach((r, i) => {
      const tot = r.reduce((a, b) => a + b, 0);
      h += `<tr>${i === 0 ? `<th rowspan="3" style="writing-mode:vertical-rl;transform:rotate(180deg)">True</th>` : ''}<th class="rowh">${CLASSES[i]}</th>`;
      r.forEach((v, j) => {
        const s = v / tot;
        const bg = i === j ? `rgba(47,158,119,${(0.12 + 0.7 * s).toFixed(2)})` : `rgba(224,82,61,${(0.05 + 0.9 * s).toFixed(2)})`;
        h += `<td style="background:${bg};color:${s > 0.55 ? '#fff' : '#0f172a'}">${num(v)}<small>${pct(s)}</small></td>`;
      });
      h += '</tr>';
    });
    return h + '</tbody></table>';
  }

  function drawCurves() {
    charts = charts.filter((c) => {
      if (['c-roc', 'c-pr'].includes(c.canvas.id)) { c.destroy(); return false; }
      return true;
    });
    const cv = D.predictions.curves;
    const g = cv.grid;
    const names = ['image', 'blood', 'text', 'fusion'];
    const line = (m, ys, label) => ({
      label, borderColor: MCOL[m], backgroundColor: MCOL[m], showLine: true, pointRadius: 0,
      borderWidth: m === 'fusion' ? 3 : 2, tension: 0,
      data: ys.map((y, i) => ({ x: g[i], y })),
    });
    mk('c-roc', {
      type: 'scatter',
      data: {
        datasets: [
          ...names.map((m) => line(m, cv.models[m].roc[curveClass].tpr, `${MNAME[m]} (AUC ${f2(cv.models[m].roc[curveClass].auc)})`)),
          { label: 'Chance', borderColor: '#94a3b8', borderDash: [5, 4], showLine: true, pointRadius: 0, borderWidth: 1, data: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
        ],
      },
      options: {
        scales: {
          x: { min: 0, max: 1, title: { display: true, text: 'False positive rate' } },
          y: { min: 0, max: 1, title: { display: true, text: 'True positive rate' } },
        },
      },
    });
    const prev = cv.models.fusion.pr[curveClass].prevalence;
    mk('c-pr', {
      type: 'scatter',
      data: {
        datasets: [
          ...names.map((m) => line(m, cv.models[m].pr[curveClass].precision, `${MNAME[m]} (AP ${f2(cv.models[m].pr[curveClass].ap)})`)),
          { label: `Base rate ${pct(prev, 1)}`, borderColor: '#94a3b8', borderDash: [5, 4], showLine: true, pointRadius: 0, borderWidth: 1, data: [{ x: 0, y: prev }, { x: 1, y: prev }] },
        ],
      },
      options: {
        scales: {
          x: { min: 0, max: 1, title: { display: true, text: 'Recall' } },
          y: { min: 0, max: 1, title: { display: true, text: 'Precision' } },
        },
      },
    });
  }

  function fusionPage() {
    const F = D.fusion;
    if (!F) return { html: pageHead('Fusion', '') + missing('Fusion results'), after() {} };
    const P = D.predictions;
    confMethod = confMethod || F.methods[0].key;
    let html = pageHead('Fusion', 'The three models each output a probability for the three classes. Fusion combines those nine numbers into one prediction. Combiners that need no training were compared against trained ones.');

    html += '<div class="grid2">' +
      card('Single model versus fusion', chartBox('c-fs', 300), { sub: 'F1 per class, validation set. The rare class gains the most.' }) +
      card('Ways of combining the models', chartBox('c-meth', 300), {
        sub: 'Validation macro F1.',
        note: 'Simple pooling beat every trained combiner. With only nine inputs and probabilities that are already reasonably calibrated, a trained combiner overfits or over-corrects toward the rare class.',
      }) + '</div>';

    html += '<div class="grid2">' +
      card('Which models are needed', chartBox('c-abl', 280), {
        sub: 'Macro F1 of the mean rule on every subset of the three models.',
        note: 'All three together beat every pair, and text contributes the most. No model is dead weight.',
      }) +
      card('Confusion matrix',
        `<select id="conf-sel">${F.methods.map((m) => `<option value="${m.key}" ${m.key === confMethod ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}</select><div id="conf-box">${confusionHTML(confMethod)}</div>`,
        { note: 'Rows are the true class, columns the prediction, and each cell also shows its share of the row.' }) +
      '</div>';

    if (P) {
      html += sectionHead('Curves', 'One-vs-rest curves for the class you pick. These use the saved validation predictions of each model and of the fusion.');
      html += `<div class="tabs" id="cls-tabs">${CLASSES.map((c) => `<button data-c="${c}" class="${c === curveClass ? 'on' : ''}">${c}</button>`).join('')}</div>`;
      html += '<div class="grid2">' +
        card('ROC curve', chartBox('c-roc', 320)) +
        card('Precision and recall', chartBox('c-pr', 320)) + '</div>';

      const cal = P.curves.models;
      const names4 = ['image', 'blood', 'text', 'fusion'];
      const bestCal = names4.slice().sort((a, b) => cal[a].calibration.ece - cal[b].calibration.ece)[0];
      // signed gap: accuracy minus stated confidence, weighted by bin size (positive = under-confident)
      const gap = (m) => {
        const bins = cal[m].calibration.bins;
        const n = bins.reduce((s, b) => s + b.n, 0);
        return bins.reduce((s, b) => s + b.n * (b.acc - b.conf), 0) / n;
      };
      const fusionSide = gap('fusion') > 0.02 ? 'under-confident' : gap('fusion') < -0.02 ? 'over-confident' : 'close to calibrated';
      html += '<div class="grid2">' +
        card('Calibration', chartBox('c-cal', 320), {
          sub: 'When a model says it is 80 percent sure, is it right about 80 percent of the time? Points on the diagonal mean yes.',
          note: `Expected calibration error: ${names4.map((m) => `${MNAME[m]} ${f2(cal[m].calibration.ece)}`).join(', ')}. ${MNAME[bestCal]} is best calibrated. The fusion is ${fusionSide}; pooling probabilities shifts them, so its confidence should be read as a ranking, not a literal probability.`,
        }) +
        card('Do the models agree?', table(['Agreement', 'Share of cases', 'Fused accuracy', 'Avg. single accuracy'],
          P.agreement.levels.map((l) => [l.name, pct(l.share), pct(l.fused_accuracy), pct(l.mean_single_accuracy)]), { numCols: [1, 2, 3] }), {
          note: 'Fused accuracy falls as the models disagree. That relationship is what the planned consistency checker and missing-information planner will build on.',
        }) + '</div>';

      html += card('How many of the three models got each true class right', chartBox('c-cnt', 230), {
        sub: 'Share of true cases of each class, by how many single models predicted it correctly.',
        note: P.agreement.per_class.map((r) => `${r.class}: fusion correct on ${pct(r.fused_correct)}`).join('. ') + `. Only ${pct(P.agreement.per_class[2].dist[2] + P.agreement.per_class[2].dist[3])} of true Lung Cancer cases are caught by two or more of the three models, so the models rarely back each other up on that class.`,
      });
    }

    if (F.coef) {
      const mx = Math.max(...F.coef.values.flat().map(Math.abs));
      const groups = ['image', 'blood', 'text'];
      let t = '<table class="heat"><thead><tr><th></th>' + groups.map((g) => `<th colspan="3">${MNAME[g]}</th>`).join('') + '</tr><tr><th></th>' +
        F.coef.cols.map((c) => `<th>${esc(c.split('.')[1].replace('_', ' '))}</th>`).join('') + '</tr></thead><tbody>';
      F.coef.values.forEach((row, i) => {
        t += `<tr><th class="rowh">${esc(F.coef.rows[i])}</th>` + row.map((v) => {
          const a = (0.08 + 0.8 * Math.abs(v) / mx).toFixed(2);
          const bg = v >= 0 ? `rgba(59,130,246,${a})` : `rgba(224,82,61,${a})`;
          return `<td style="background:${bg};color:${Math.abs(v) / mx > 0.6 ? '#fff' : '#0f172a'}">${v.toFixed(2)}</td>`;
        }).join('') + '</tr>';
      });
      html += card('What a trained combiner learned', t + '</tbody></table>', {
        sub: 'Class-balanced logistic stack. Each row is an output class; each column is a probability from one model.',
        note: 'Blue raises the class score, red lowers it. The largest weights sit on each class\'s own probability. These weights do not show how much each model helps, because they depend on how wide each model\'s probability range is. Use the ablation and the curves above for that.',
      });
    }

    const FM = D.fusion_mlp;
    if (FM) {
      html += sectionHead('An alternative: fusing the features instead of the predictions',
        'The models above combine each modality\'s predicted probabilities (late fusion). This instead concatenates the three modalities\' raw feature vectors into one and trains a single MLP on that, to see whether combining earlier does better.');
      html += '<div class="grid2">' +
        card('Feature fusion versus late fusion', chartBox('c-mlpvs', 230), {
          note: `Feature fusion reaches ${f2(FM.final.macro_f1)}, below late fusion\'s ${f2(FM.late_fusion_bar)}. Its own held-out tuning set scored ${f2(Math.max(...Object.values(FM.ablation).map((a) => a.dev)))} with the same model, so the drop on validation is overfitting: ${FM.final.n_features} input features against only ${num(FM.n_fit)} training rows (and far fewer of the rare Lung Cancer class) is enough for a trained model to fit patterns that do not carry over. Late fusion is what the project uses going forward.`,
        }) +
        card('Same modality pattern as late fusion', chartBox('c-mlpabl', 230), {
          sub: 'Validation macro F1, feature-level fusion.',
          note: 'All three modalities beat every pair here too, and text still contributes the most.',
        }) + '</div>';
    }

    const after = () => {
      const sets = ['image', 'blood', 'text'].map((m) => ({
        label: MNAME[m], backgroundColor: MCOL[m], borderRadius: 4,
        data: [...CLASSES.map((c) => F.single[m].f1_per_class[c]), F.single[m].macro_f1],
      }));
      sets.push({ label: 'Fusion', backgroundColor: MCOL.fusion, borderRadius: 4, data: [...CLASSES.map((c) => F.methods[0].f1[c]), F.methods[0].macro_f1] });
      classBar('c-fs', sets, { labels: [...CLASSES, 'Macro F1'], options: { plugins: { valueLabels: { format: () => '' } } } });

      const ms = F.methods.slice();
      mk('c-meth', {
        type: 'bar',
        data: {
          labels: ms.map((m) => m.label),
          datasets: [{ data: ms.map((m) => m.macro_f1), backgroundColor: ms.map((m) => (m.kind === 'pooled' ? '#4f46e5' : '#cbd5e1')), borderRadius: 5 }],
        },
        options: {
          indexAxis: 'y', layout: { padding: { right: 40 } },
          plugins: { legend: { display: false }, valueLabels: { format: (v) => Number(v).toFixed(3) } },
          scales: { x: { min: 0.4, max: 0.7, title: { display: true, text: 'Macro F1' } }, y: { grid: { display: false } } },
        },
      });

      const ab = Object.entries(F.ablation).sort((a, b) => a[1] - b[1]);
      mk('c-abl', {
        type: 'bar',
        data: {
          labels: ab.map(([k]) => k.split('+').map((s) => MNAME[s]).join(' + ')),
          datasets: [{ data: ab.map(([, v]) => v), backgroundColor: ab.map(([k]) => (k.split('+').length === 3 ? '#0f172a' : k.includes('+') ? '#818cf8' : '#cbd5e1')), borderRadius: 4 }],
        },
        options: {
          indexAxis: 'y', layout: { padding: { right: 40 } },
          plugins: { legend: { display: false }, valueLabels: { format: (v) => Number(v).toFixed(3) } },
          scales: { x: { min: 0.4, max: 0.65, title: { display: true, text: 'Macro F1' } }, y: { grid: { display: false } } },
        },
      });

      const sel = document.getElementById('conf-sel');
      if (sel) sel.addEventListener('change', () => { confMethod = sel.value; document.getElementById('conf-box').innerHTML = confusionHTML(confMethod); });

      if (P) {
        drawCurves();
        document.querySelectorAll('#cls-tabs button').forEach((b) => b.addEventListener('click', () => {
          curveClass = b.dataset.c;
          document.querySelectorAll('#cls-tabs button').forEach((x) => x.classList.toggle('on', x === b));
          drawCurves();
        }));

        const cal = P.curves.models;
        mk('c-cal', {
          type: 'scatter',
          data: {
            datasets: [
              ...['image', 'blood', 'text', 'fusion'].map((m) => ({
                label: MNAME[m], borderColor: MCOL[m], backgroundColor: MCOL[m], showLine: true, borderWidth: 2, pointRadius: 4,
                data: cal[m].calibration.bins.map((b) => ({ x: b.conf, y: b.acc })),
              })),
              { label: 'Perfect', borderColor: '#94a3b8', borderDash: [5, 4], showLine: true, pointRadius: 0, borderWidth: 1, data: [{ x: 0.3, y: 0.3 }, { x: 1, y: 1 }] },
            ],
          },
          options: {
            scales: {
              x: { min: 0.3, max: 1, title: { display: true, text: 'Stated confidence' } },
              y: { min: 0, max: 1, title: { display: true, text: 'Actual accuracy' } },
            },
          },
        });

        const shades = ['#e0523d', '#f0a35e', '#9bd3b6', '#2f9e77'];
        mk('c-cnt', {
          type: 'bar',
          data: {
            labels: P.agreement.per_class.map((r) => `${r.class} (n=${r.n})`),
            datasets: [0, 1, 2, 3].map((j) => ({
              label: `${j} of 3 correct`, backgroundColor: shades[j],
              data: P.agreement.per_class.map((r) => r.dist[j]),
            })),
          },
          options: {
            indexAxis: 'y',
            scales: { x: { stacked: true, max: 1, title: { display: true, text: 'Share of cases' } }, y: { stacked: true, grid: { display: false } } },
          },
        });
      }

      if (FM) {
        mk('c-mlpvs', {
          type: 'bar',
          data: {
            labels: ['Late fusion (logpool)', 'Feature fusion (MLP)'],
            datasets: [{ data: [FM.late_fusion_bar, FM.final.macro_f1], backgroundColor: [MCOL.fusion, '#94a3b8'], borderRadius: 6 }],
          },
          options: {
            plugins: { legend: { display: false }, valueLabels: { format: (v) => Number(v).toFixed(3) } },
            scales: { y: { min: 0, max: 0.7, title: { display: true, text: 'Validation macro F1' } }, x: { grid: { display: false } } },
          },
        });
        const ak = Object.keys(FM.ablation);
        mk('c-mlpabl', {
          type: 'bar',
          data: {
            labels: ak,
            datasets: [
              { label: 'Dev (tuning set)', backgroundColor: '#cbd5e1', borderRadius: 4, data: ak.map((k) => FM.ablation[k].dev) },
              { label: 'Val', backgroundColor: '#0f172a', borderRadius: 4, data: ak.map((k) => FM.ablation[k].val) },
            ],
          },
          options: {
            indexAxis: 'y',
            scales: { x: { min: 0, max: 0.7, title: { display: true, text: 'Macro F1' } }, y: { grid: { display: false } } },
          },
        });
      }
    };
    return { html, after };
  }

  /* ------------------------------------------------------------ explain */
  async function explainPage() {
    let items = [];
    try {
      const r = await fetch('/api/gallery');
      items = (await r.json()).items || [];
    } catch (e) { items = []; }

    let html = pageHead('Explainability',
      'Every prediction comes with an explanation per model: Grad-CAM on the X-ray, SHAP values for the labs, and word-level attribution for the clinical indication. These are real validation cases run through the explainers.');

    if (!items.length) {
      return { html: html + card('', '<p class="muted">No sample explanations found. Run <code>python -m explain.run --study-id ID --save experiments/explain_samples/</code> for a few validation cases.</p>'), after() {} };
    }

    html += '<div class="callout">Red highlights and bars push toward the model\'s predicted class; blue pushes away. Each case shows one model per column, with a green top edge when that model got the true class right and a red one when it did not.</div>';

    items.forEach((it, idx) => {
      const heads = ['image', 'blood', 'text'].filter((m) => it.modalities[m]);
      html += `<div class="case"><div class="case-head"><h3>Case ${idx + 1}</h3><span class="badge" style="background:${CCOL[it.true_label] || '#64748b'}">True label: ${esc(it.true_label)}</span><span class="muted" style="font-size:12px">study ${esc(it.study_id)}</span></div><div class="grid3">`;
      heads.forEach((m) => {
        const r = it.modalities[m];
        const ok = r.predicted_class === it.true_label;
        const probs = CLASSES.map((c) => {
          const p = r.probabilities[c] || 0;
          return `<div class="prob"><span>${c}</span><span class="bar"><i style="width:${(p * 100).toFixed(0)}%;background:${CCOL[c]}"></i></span><span>${pct(p)}</span></div>`;
        }).join('');
        const chips = r.top.slice(0, 5).map((t) => {
          const v = t.value != null && t.value !== '' ? ` ${esc(t.value)}` : '';
          return `<span class="chip ${t.effect > 0 ? 'up' : 'down'}">${esc(t.feature)}${v} ${t.effect > 0 ? '+' : ''}${Number(t.effect).toFixed(2)}</span>`;
        }).join('');
        html += `<div class="mod ${ok ? 'right' : 'wrong'}"><h4>${MNAME[m]}<span class="badge" style="background:${CCOL[r.predicted_class]}">${esc(r.predicted_class)}</span></h4>` +
          (r.image ? `<a href="${r.image}" target="_blank" rel="noopener"><img src="${r.image}" alt="${MNAME[m]} explanation" loading="lazy"></a>` : '') +
          `<p class="summary">${esc(r.summary)}</p><div class="probs">${probs}</div><div class="chips">${chips}</div></div>`;
      });
      html += '</div></div>';
    });
    return { html, after() {} };
  }

  /* ------------------------------------------------------------ roadmap */
  function roadmapPage() {
    const phases = D.phases || [];
    let html = pageHead('Roadmap', 'Where each phase stands, read from ROADMAP.md, and what is left.');
    html += card('Phases', table(['Phase', 'Title', 'State', 'Scope'],
      phases.map((p) => [String(p.num), esc(p.title), pill(p.state), esc(p.blurb)])));
    html += card('Remaining work', table(['Phase', 'What it involves', 'Rough effort'],
      (D.remaining || []).map((r) => [`<b>${esc(r.phase)}</b>`, esc(r.work), esc(r.effort)])), {
      note: 'Effort is a rough estimate in focused working days, not a commitment.',
    });
    html += card('What this dashboard will gain',
      `<p><b>Analyze a case.</b> Upload a chest X-ray, type the clinical indication, and enter any blood labs (all optional). The page will show the fused prediction, the agent's step-by-step belief updates, any disagreement between the models, per-model explanations, and cited evidence passages.</p>
       <p>The pages above stay as they are and update when new results are produced.</p>`);
    return { html, after() {} };
  }

  /* ------------------------------------------------------------ router */
  const routes = { overview, data: dataPage, models: modelsPage, fusion: fusionPage, explain: explainPage, roadmap: roadmapPage };

  async function route() {
    const page = location.hash.replace('#/', '') || 'overview';
    const fn = routes[page] || overview;
    charts.forEach((c) => c.destroy());
    charts = [];
    document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.page === (routes[page] ? page : 'overview')));
    const out = await fn();
    main.innerHTML = out.html;
    out.after();
    window.scrollTo(0, 0);
  }

  async function init() {
    try {
      const r = await fetch('/api/dashboard');
      if (!r.ok) throw new Error(r.status);
      D = await r.json();
    } catch (e) {
      main.innerHTML = '<section class="card"><h3>Dashboard data not available</h3><p class="muted">Run <code>python -m app.dashboard_data</code> from the repo root, then reload.</p></section>';
      return;
    }
    window.addEventListener('hashchange', route);
    route();
  }

  init();
})();
