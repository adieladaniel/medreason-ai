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
  (`densenet121-res224-mimic_ch`), its 18 pathology-probability outputs feed a small MLP
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
  embed-then-logreg pipeline. Per-word effect on P(predicted). About 60 to 90 seconds per
  case on CPU (`max_evals=300`). Renders the indication with words shaded by effect.
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

Status: not started.

- Agent orchestrator: a tool-calling shell around the trained models (image, blood, text
  tools plus fusion).
- Belief state: probability distribution over the 3 classes, updated per tool call.
- Consistency checker: cross-modality conflict detection (for example image says Cancer,
  blood and text say Normal).
- Missing-information planner: decide whether more input is worth requesting given current
  confidence.
- Design rule to preserve: the agent orchestrates and reasons, the trained models
  diagnose.

## Phase 6, evidence retrieval (RAG)

Status: not started.

- Build a small corpus of relevant papers and guidelines for the 3 target diseases.
- Qdrant vector DB for retrieval.
- Evidence planner: build a query from the prediction, the explainability output, and the
  belief state, not just the raw label.
- Local LLM via Ollama for the final explanation text. RAG explains, it does not diagnose.

## Phase 7, backend and API

Status: not started. A first slice exists: the progress dashboard (`app/`), a FastAPI app
with a static single-page UI that shows all results so far and rebuilds its data from
`experiments/` automatically. The pipeline endpoints and the Analyze page will be added
to it.

- FastAPI service wrapping the full pipeline: upload, prediction, belief state, consistency
  check, explanation, evidence.
- PostgreSQL for case and result storage.
- Docker containerization of the full stack.

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
