# MedReason-AI project roadmap

Living document tracking where the project stands and what is left. Update the status
lines as phases complete. Writing style: plain prose, no em dashes (see CLAUDE.md
section 6).

---

## Phase 0, vision and scope

Status: done.

- Defined the project as an agentic multimodal medical diagnostic assistant (image, blood,
  text), not just a classifier.
- In scope: prediction, explainability, agentic orchestration, RAG evidence retrieval. Out
  of scope: treatment and dosage, autonomous decisions, clinical deployment.
- Target diseases finalized empirically at Normal, Pneumonia, Lung Cancer. Tuberculosis
  and COVID-19 were dropped, see the Phase 1 findings.

## Phase 1, data foundation

Status: done.

- Selected MIMIC-CXR (Kaggle mirror) and MIMIC-IV v3.1 hospital module as source datasets.
- Built and empirically validated the linking algorithm (subject_id, then study_datetime,
  then admission window, then hadm_id). About 48% of CXR studies link to a valid admission.
- Found and handled real data constraints. COVID-19 is structurally absent because the
  imaging predates the pandemic. Tuberculosis is numerically negligible (10 cases). Both
  were dropped from scope.
- Built a lightweight negation-aware report labeler (spaCy + negspacy) to validate labels
  against the radiology report text, not just ICD codes.
- Checked blood-test tier coverage empirically. CRP and ESR are too sparse (3.7% and 7.5%)
  to use as required features.
- Assembled the final joined dataset: 15,453 samples (Pneumonia 10,165 / Normal 4,277 /
  Lung Cancer 1,011), subject-level stratified train/val/test split (11,039 / 2,315 /
  2,099).
- Output: `data/processed/{final_dataset,train,val,test}.csv`.

## Phase 2, unimodal baselines

Status: done. Image 0.50 / blood 0.46 / text 0.51 (val macro F1). The modalities are
complementary.

- Image baseline v1: frozen ImageNet-pretrained DenseNet121 + linear head, 8 epochs. Macro
  F1 0.45 (Pneumonia F1 0.73, Normal 0.49, Cancer F1 0.12, precision 0.09).
- Image baseline v2: unfroze the last dense block, discriminative LR + cosine schedule, 20
  epochs. Macro F1 0.46 (Pneumonia 0.71, Normal 0.52, Cancer 0.14, precision about 0.10).
  Marginal gain only. Cancer precision stayed stuck, which points to a feature-quality
  bottleneck (generic ImageNet features), not an undertraining problem.
- Image baseline v3: frozen `torchxrayvision` DenseNet121 pretrained directly on MIMIC-CXR
  (`densenet121-res224-mimic_ch`), its 18 pathology-probability outputs (11 trained, 7 constant) feed a small MLP
  head, 15 epochs. Macro F1 0.50 (Pneumonia 0.77, Normal 0.56, Cancer 0.17, precision
  0.14). Clear improvement across every class over v1 and v2. Current best baseline.
- Blood/tabular baseline: `scripts/train_blood_baseline.py`. sklearn
  `HistGradientBoostingClassifier` (native NaN handling) on the 17 lab columns plus NLR and
  PLR derived ratios, `class_weight="balanced"`. Macro F1 0.46 (Pneumonia 0.73, Normal
  0.45, Cancer 0.20). Beats logreg (0.40) and the majority-class floor (0.27). Most
  informative labs: hemoglobin, albumin, LDH, WBC, lymphocytes. CRP and ESR are near
  useless, as their 5 to 9% coverage predicted. Blood alone is slightly weaker than image
  v3 overall but marginally better on the Cancer class. Restricting to rows with at least
  one core lab does not help (0.456), so missingness is not the main limiter. Blood just
  carries limited signal for this 3-way split. Artifacts and val predicted probabilities
  in `experiments/blood_baseline/`.
- Text baseline: `scripts/train_text_baseline.py`. Frozen Bio_ClinicalBERT (mean-pooled) +
  logistic-regression head. Two settings, because the labels are partly report-derived
  (Pneumonia 65% / Cancer 54% / Normal 100% depend on report text). Setting A, INDICATION
  section only (the referring clinician's reason-for-exam, never used in labeling, the
  honest number): macro F1 0.51 (Normal 0.53, Pneumonia 0.69, Cancer 0.32). Setting B,
  full report: macro F1 0.68 but leakage-inflated, not a capability estimate. Leakage
  analysis: on the ICD-only val subset (where the report cannot leak) A and B are about
  equal (0.30 vs 0.33), so B's lead is entirely the report-derived subset. Text is the
  best unimodal for the Cancer class (0.32 vs image 0.17 / blood 0.20). Artifacts and both
  settings' val probabilities in `experiments/text_baseline/`.
- Takeaway: image 0.50 / blood 0.46 / text 0.51, all similar overall but with different
  per-class strengths (image on Pneumonia, text on Cancer), so genuinely complementary.
  Fusion (Phase 3) has an empirical case.
- Cancer-class decision (post-fusion): fusion lifted Cancer F1 from 0.32 to 0.43 with no
  special handling. Further gains (focal loss, cascade, threshold tuning) are deferred. Not
  worth the added complexity right now given the improvement. Revisit only if the final
  test-set numbers demand it.

## Phase 3, multimodal fusion

Status: done (late fusion). Fused val macro F1 0.61 vs best single modality 0.51 (+0.10).

- Late (decision-level) fusion: stack the three Phase 2 baselines' 3-class probability
  vectors (9 features) into a final label. `scripts/train_fusion.py`.
- OOF plumbing: `scripts/generate_oof_predictions.py` (blood and text, 5-fold
  subject-grouped StratifiedGroupKFold) and the Kaggle kernel
  `medreason-ai-image-oof-val-predictions` (image, same folds) produce out-of-fold train
  predictions so the meta-model is not fit on the base models' own training rows. Val
  predictions are saved per modality too.
- Result (val): logpool (geometric mean) 0.613, mean-prob 0.600, unweighted logreg stack
  0.585, balanced logreg 0.562, hgb stack 0.561. Parameter-free pooling beats the trained
  stackers, since with only 9 inputs and well-calibrated base probabilities the trained
  meta-models overfit or over-correct. Per class: Normal 0.53 to 0.62, Pneumonia 0.69 to
  0.80, Cancer 0.32 to 0.43 (the biggest relative lift, the weak class benefits most).
- Ablation (mean rule): all three (0.60) beat the best pair blood+text (0.576), which
  beats image+text (0.557), which beats image+blood (0.493). Text contributes most, image
  least.
- Leakage guard: text inputs are Setting A (INDICATION section only), never full-report.
- Artifacts in `experiments/fusion/`.

Feature-level fusion was tried as a documented ablation
(`scripts/train_fusion_mlp.py`, done 2026-09-20): concatenate the three modalities'
frozen representations (image's 18 pathology probabilities, blood's 19 features
imputed, text's 768-dim embedding reduced to 32 by PCA) and train one MLP on the
result, instead of stacking predicted probabilities. No out-of-fold step is needed,
since these are frozen encoder outputs that never saw the labels. Result: val macro F1
0.586, below late fusion's 0.613. The dev set used for early stopping scored 0.655 on
the same model, so the gap is overfitting: 88 features and only 774 cancer training
rows is enough for a trained model to fit patterns that do not generalize. The same
modality ablation held here too (all three beats every pair, text contributes most,
image least). Late fusion stays the one used going forward. Artifacts in
`experiments/fusion_mlp/`.

## Phase 4, explainability

Status: core done. `explain/` package, one module per modality. Each loads a saved model
and returns a common contract (predicted class, probabilities, ranked signed attributions,
one-line summary) plus a matplotlib figure.

- Blood (`explain/blood.py`): SHAP `TreeExplainer` (tree_path_dependent, exact) on the
  saved `HistGradientBoostingClassifier`. Attributions in log-odds, horizontal bar plot.
- Text (`explain/text.py`): SHAP with a word-level (`\W+`) Text masker over the whole
  embed-then-logreg pipeline. Per-word effect on P(predicted). 0.4 to 0.9 s for a typical
  indication and 10 to 17 s for a 100-word one (measured; an earlier 60 to 90 s figure did not
  reproduce). One-word or empty indications get an exact single-word effect or none. Renders the indication with words shaded by effect.
- Image (`explain/image.py`): Grad-CAM. Keep image, backbone, 18 pathology probs, and the
  3-class logit as one differentiable chain, hook the backbone's last conv map, backprop
  the predicted logit (the input needs `requires_grad` because the backbone is frozen).
  Also a pathology-channel attribution (`d(logit)/d(prob) * prob`) that names which of the
  18 xrv channels drove the prediction.
- `explain/run.py`: unified entrypoint. Runs all three for one `study_id`, prints, and
  saves PNGs plus JSON. This is what the Phase 5 agent will call.
- Decoupled from training code (feature and preprocessing logic duplicated by hand, kept
  in sync).
- Local quirk fixed: `import torch` before sklearn or shap, or torch's DLL init fails on
  Windows.
- Verified on one confident-correct case per class plus two cross-modality-conflict cases
  (text and blood right, image wrong on Cancer). Explanations are clinically coherent.
- Deferred to the UI and demo phases: a combined one-page per-case view, figure polish.

Core explainability capability is complete. The agent (Phase 5) can consume
`explain/run.py`.

## Phase 5, agentic layer

Status: core done, 2026-09-20. `app/agent/` package.

- Tools (`tools.py`): thin wrappers giving `explain.image/blood/text` a single calling
  convention. No new prediction logic; every probability still comes from the saved
  Phase 2 models.
- Belief state (`belief.py`): pools whichever modalities have run, using the exact rule
  validated in Phase 3, the geometric mean (`scripts/train_fusion.py`, logpool, val macro
  F1 0.613). Recomputed from scratch on every update rather than folded in as a running
  product, so the belief after any subset of tools matches that subset's row in the Phase
  3 ablation table exactly.
- Consistency checker (`consistency.py`): flags disagreement between modalities. The
  confidence language it uses ("cases like this are correct N% of the time") is read
  from real validation statistics in `app/data/dashboard.json`, with the same numbers as
  a fallback constant if that file is missing.
- Missing-information planner (`planner.py`): decides whether another modality is worth
  requesting. Confidence thresholds (0.85 high, 0.55 low) are a design choice, but which
  modality it recommends comes from the real Phase 3 ablation ranking (text contributes
  most, then blood, then image).
- Orchestrator (`orchestrator.py`): runs the above for one case. CLI mirrors
  `explain/run.py`: `python -m app.agent.orchestrator --study-id ID [--drop image|blood|text]`,
  the `--drop` flags simulate a missing modality, which is also how the Phase 8b demo
  cases (missing labs, cross-modality conflict) will be produced.
- Verified against real validation cases: full agreement, a missing modality, a genuine
  cross-modality conflict (text and blood correctly call Lung Cancer, image confidently
  calls Normal, the checker flags it and the pooled belief still lands on Cancer), and a
  single-modality low-confidence case that correctly recommends text next.
- Design rule preserved: the orchestrator calls tools and reasons about their outputs, it
  never produces a class prediction of its own.
- Not yet wired to an API or the UI. The dashboard's "Analyze a case" is still a
  placeholder; that wiring is Phase 7 work.

## Phase 6, evidence retrieval (RAG)

Status: done. Corpus and retrieval core done 2026-10-11.

- Corpus (done): 44 open-access papers (27 at first, 17 added to close coverage gaps), about 144,500 words, listed in
  `app/rag/corpus_sources.json` and downloaded by `python -m app.rag.build_corpus` from
  Europe PMC into `data/corpus/` (git-ignored). The second batch of 13 was chosen from the
  models' own top features and image channels. Every license was checked against Europe PMC
  (CC BY, CC BY-NC, CC BY-NC-SA; CC BY-NC-ND excluded because chunking is a derivative use).
  Only the manifest and the script are committed.
- Retrieval (done, measured): hybrid search (embeddings plus BM25), then a biomedical
  cross-encoder reranks the top 5, then a score threshold lets the system say "no supporting
  evidence" (`app/rag/retrieve.py`). Evaluated on 54 queries split into dev and test plus 16
  off-corpus queries (`experiments/rag_retrieval/PIPELINE.md`). Right passage at rank 1 went
  from 0.73 / 0.57 (dev / test) for the semantic-only start to 1.00 / 0.86, and the threshold
  refused all 16 off-corpus queries. Keyword search alone (0.81 / 0.79) beat the semantic
  model, query expansion did not help and was dropped, and a hand audit of 18 results
  (`AUDIT.md`) shows the keyword-based relevance proxy is roughly right but approximate.
  Retrieval takes about 1 second per query on this CPU when the laptop is quiet and about 4 seconds with
  a browser open (measured, `eval_latency.py`, p50 1.1 s and p95 1.3 s on a quiet run), almost all of it
  the reranker. Under heavy load it was 16 seconds.
- Known limits: a reranker can rank a near-miss first with a high score, and a high score
  means on-topic rather than supportive, so the explanation writer must check what a passage
  actually states.
- Evidence planner (done, `app/rag/planner.py`): one keyword query per real driver (labs outside
  their reference range, strong image channels, meaningful indication words, and a
  differential query when modalities disagree), not one query from the label. A Normal
  prediction requests nothing.
- Explanation writer (done, `app/rag/writer.py`): a local model through Ollama. Sentences about
  the case are built by code, and the model only summarises one retrieved passage at a time.
  A first version that gave the model the whole case invented citations, which is why.
- Faithfulness check (done, `app/rag/faithfulness.py`): a sentence is shown only if an NLI
  model finds it entailed by its passage, its numbers appear in the passage, and it contains no
  advice wording. Checked on 35 hand-made sentences (`eval_faithfulness.py`); small, so it is a
  sanity check. RAG explains, it does not diagnose, and none of this can change the label.
- Topic gate (done): a written sentence must mention the finding its query was about, because
  the writer model never declined a passage on its own and a faithful sentence can still be
  misleading context.
- Writer comparison and expansion (done, `eval_writer.py`): llama3.2 3B beat Qwen2.5 3B on the
  share of sentences passing the check (73% against 63%, 41 items), and showing the writer the
  neighbouring passage did not help, so neither Qwen nor parent-child expansion is used.
- Dashboard Evidence page (done).
- Claim checker comparison (done, `eval_checker.py`): the current NLI checker against two MiniCheck
  models on 74 labelled sentences. MiniCheck is faster and drops fewer faithful sentences but
  caught none of the three subtle real errors, so the NLI checker stays. No checker catches
  sentences that blend true facts into a false claim, so the interface must show background as
  what a cited paper says.
- Corpus coverage (done, `eval_gaps.py`): the planner's 52 possible queries were all run through
  retrieval and 29 had no passage that mentions the finding. 17 more papers (44 in all, 811
  passages) and a second query phrasing for image findings cut that to 13, and the 54-query
  retrieval evaluation did not regress (test right-passage-first 0.857 to 0.893, off-corpus
  queries still refused 16 of 16). The remaining gaps are mostly topics with little supportive
  literature, so "no passage" stays the honest answer.
- Carried to Phase 7: a faster checker (it takes about 6 s per sentence on CPU).

## Reliability checks (done on validation, repeat once on test)

- `scripts/eval_uncertainty.py`: subject-level bootstrap intervals, calibration with temperature
  scaling, selective prediction, class-conditional conformal sets, and subgroup results by sex,
  age and recorded race. The fusion gain over the best single model is +0.10 (95% interval +0.07
  to +0.13). Probabilities were under-confident and are now close to calibrated. Lung Cancer is
  declined more often than other classes by a confidence rule, which matters for any abstention
  design. The final test-set run should repeat all of this once, with nothing re-tuned.

- Text dominance check (done, `scripts/eval_dominance.py`): text is the strongest input and is close to a keyword rule
  (0.502 against 0.509), image and blood each add a small but clearly nonzero amount, text dominates only the rare class,
  and averaging the models loses cancers the text model finds (50 percent against 61 percent). Options recorded in
  CLAUDE.md. Prediction sets, a low-confidence headline and a cancer alert were built on 2026-10-26 (`app/agent/decision.py`,
  rules from `scripts/build_decision_rules.py`); better image weights remain open.

## Phase 7, backend and API

Status: in progress. The design and its reasons are in `docs/PHASE7_DESIGN.md`. The progress
dashboard (`app/`) already exists and the Analyze page will be added to it.

Steps, built and reported one at a time:

1. Storage (done 2026-10-14): `app/config.py`, `app/database/{models,session,repository}.py`.
   Cases, events and language-model calls, SQLite by default and Postgres through one setting,
   restart recovery, retention, tested on SQLite (12 tests).
2. Staged pipeline and job runner (done 2026-10-25): `app/pipelines/`. The prediction is stored
   first (0.3 s warm, 6 s cold), then explanations, then retrieval, writing and verifying (about 35 s
   warm, nearly all the language model and the entailment checker). One in-process worker, results saved
   stage by stage, a late failure leaves a "partial" case, a retrieval cache, and a belief identical to
   `run_case` on real cases.
3. API (done 2026-10-25): `app/api/`. `POST /api/cases` answers 202, then status, results, an audit trail,
   a live event stream, figures and the uploaded image are readable; upload and input checks, a queue limit,
   startup recovery and expiry. Run against the real server and models. No login, and no check that an image is
   a chest X-ray.
4. Analyze page in the dashboard (done 2026-10-25): `app/static/analyze.js`. A form built from the API's own input
   spec, a live stage tracker, the prediction with a low-confidence flag, per-input explanations, the cited evidence
   with each passage one click away, an audit view, recent cases and delete. Tested in a real browser (14 tests)
   and run once with the real models.
5. Observability: structured logs with case ids, a readiness check for database, Qdrant and Ollama.
6. Containers: Dockerfile, compose (API, Postgres, Qdrant), Qdrant server mode, a CI workflow that
   runs the tests. Docker Desktop 4.49.0 is installed and verified (2026-10-25, notes in
   `docs/ENVIRONMENT_AND_HANDOFF.md`). Ollama stays on the Windows host and the container's
   Postgres uses host port 5433. Done ahead of the compose file: the Ollama and Qdrant addresses are
   settings (`app/config.py`), Qdrant server mode and the storage layer on Postgres were each run
   against a real container, and `requirements.txt` has the missing packages.
7. End-to-end run on real cases and a timing budget.

## Phase 8, integration, testing, documentation

Status: not started.

- End-to-end testing across the full pipeline.
- Write up the methodology and empirical findings (dataset linking rate, label validation
  approach, class imbalance handling, per-modality results). This doubles as thesis
  material.
- Final polish pass.

### Two kinds of evaluation

- Quantitative (the thesis number): one batch run of the full pipeline over all 2,099 rows
  of `data/processed/test.csv`. The file already has `image_path`, `report_path`, the 17
  blood columns, and `final_label`. The script resolves the image, extracts the INDICATION
  section from the report (same `extract_indication` as training, so the leakage rule
  holds), runs image, blood, text, fusion, and agent, and compares to `final_label`.
  Report macro F1, per-class F1, confusion matrix, calibration, plus agent-layer stats
  (how often the consistency checker fires, how often it escalates or requests more data).
  `test.csv` stays sealed until this run. Develop Phases 4 to 7 against `val.csv`.
- Interactive demo (the viva): the UI, one case at a time.

### Phase 8b, demo preparation for the final-year defense

- Case-curation script: scan `val.csv` (never `test.csv`) for cases matching each
  edge-case profile, produce a vetted shortlist of about 15 study IDs with notes, then
  freeze about 8 to 12 for the presentation into `demo/cases/` (git-ignored, MIMIC
  images). Profiles to hit: clean Normal, Pneumonia, and Cancer; cross-modality conflict
  (modalities disagree and the consistency checker fires); low confidence (fused max prob
  around 0.4 to 0.55, triggers the missing-info planner); no blood labs provided (graceful
  degradation); model wrong but flags it (low confidence or a fired conflict, which is a
  strength to show, not hide); non-CXR or garbage image (needs a lightweight OOD input
  guard, build one).
- UI: a "load example case" dropdown that populates the form from `demo/cases/`.
- Inputs a user provides: a frontal chest X-ray (PNG or JPG); the clinical indication or
  reason-for-exam as short free text, not a radiology report (leakage-safe and realistic,
  since pre-read you have the film, the labs, and the referral reason); up to 17 blood
  labs, all optional, with a "no bloods" checkbox; age and sex, optional.
- Framing for modest accuracy (fusion about 0.61 macro F1): this is a systems-engineering
  project (agentic pipeline, explainability, belief state, consistency checking, RAG, real
  backend), not a benchmark-accuracy project. Demo the reasoning and explanations working.
  A case the model gets wrong but the consistency checker catches is a highlight.
- De-risk before the defense:
  - Offline-first. All model weights vendored locally, zero runtime downloads.
  - Ollama latency on 4 GB VRAM: use a small quantized model (Qwen2.5-3B or Phi-3),
    pre-warm it, stream the RAG text after showing the prediction and explainability, and
    pre-cache explanations for the frozen demo cases.
  - The serving path preprocessing must match the torchxrayvision training exactly
    (grayscale, normalize to about [-1024, 1024], center-crop, resize 224). Verify against
    known cases.
  - Rehearse the Docker cold start (Postgres, Qdrant, Ollama, API). Have a reset script to
    clear the case table between runs.
  - Record a full screen-capture of the demo end to end as a backup for projector or
    laptop failure.
- Data provenance: get PhysioNet credentials before the defense (free, about 1 to 2 weeks,
  supervisor as reference). This closes the "is your data use authorized" question.

---

## Where things run

- Data pipeline and exploration scripts run locally (`scripts/`, `scripts/exploration/`).
  Lightweight, no GPU needed.
- Model training runs on Kaggle Kernels (free GPU), not on the local laptop. See CLAUDE.md
  section 4.
- Backend, agent, and API (future phases) will run locally in VS Code, with Docker for
  Postgres and Qdrant and Ollama as a local service.
