# MedReason-AI — Project Roadmap

Living document tracking where the project stands and what's left. Update the status markers as phases complete.

Legend: ✅ done · 🔄 in progress · ⬜ not started

---

## Phase 0 — Vision & Scope
✅ Done
- Defined project as an agentic multimodal medical diagnostic assistant (image + blood + text), not just a classifier.
- Scoped in: prediction, explainability, agentic orchestration, RAG evidence retrieval. Scoped out: treatment/dosage, autonomous decisions, clinical deployment.
- Target diseases finalized empirically at **Normal / Pneumonia / Lung Cancer** (Tuberculosis and COVID-19 dropped — see Phase 1 findings).

## Phase 1 — Data Foundation
✅ Done
- Selected MIMIC-CXR (Kaggle mirror) + MIMIC-IV v3.1 hospital module as source datasets.
- Built and empirically validated the subject_id → study_datetime → admission-window → hadm_id linking algorithm (~48% of CXR studies link to a valid admission).
- Discovered and handled real data constraints: COVID-19 structurally absent (imaging predates the pandemic), Tuberculosis numerically negligible (10 cases) — both dropped from scope.
- Built a lightweight negation-aware report labeler (spaCy + negspacy) to validate labels against actual radiology report text, not just ICD codes.
- Checked blood-test tier coverage empirically — CRP/ESR found to be too sparse (3.7%/7.5%) to use as required features.
- Assembled final joined dataset: **15,453 samples** (Pneumonia 10,165 / Normal 4,277 / Lung Cancer 1,011), subject-level stratified train/val/test split (11,039 / 2,315 / 2,099).
- Output: `data/processed/{final_dataset,train,val,test}.csv`.

## Phase 2 — Unimodal Baselines
✅ Done — image 0.50 / blood 0.46 / text 0.51 (val macro F1), modalities are complementary
- ✅ **Image baseline v1**: frozen ImageNet-pretrained DenseNet121 + linear head, 8 epochs. Result: macro F1 0.45 (Pneumonia F1 0.73, Normal 0.49, **Cancer F1 0.12, precision 0.09**).
- ✅ **Image baseline v2**: unfroze last dense block, discriminative LR + cosine schedule, 20 epochs. Result: macro F1 0.46 (Pneumonia 0.71, Normal **0.52**, Cancer **0.14**, precision still ~0.10). Marginal gain only — Cancer precision stuck, pointing to a feature-quality bottleneck (generic ImageNet features), not an undertraining problem.
- ✅ **Image baseline v3**: frozen `torchxrayvision` DenseNet121 pretrained directly on MIMIC-CXR (`densenet121-res224-mimic_ch`), its 18 pathology-probability outputs feed a small MLP head, 15 epochs. Result: macro F1 **0.50** (Pneumonia 0.77, Normal 0.56, Cancer **0.17**, precision 0.14) — clear improvement across every class over v1/v2. **Current best baseline.**
- ⬜ Next: either unfreeze part of this MIMIC-CXR backbone for further gains, address remaining Cancer weakness (oversampling/focal loss), or move on to the blood/text unimodal baselines below.
- ✅ **Blood/tabular baseline**: `scripts/train_blood_baseline.py` — sklearn `HistGradientBoostingClassifier` (native NaN handling) on the 17 lab columns + NLR/PLR derived ratios, `class_weight="balanced"`. Result: macro F1 **0.46** (Pneumonia 0.73, Normal 0.45, **Cancer 0.20**). Beats logreg (0.40) and the majority-class floor (0.27). Most informative labs: hemoglobin, albumin, LDH, WBC, lymphocytes; CRP/ESR near-useless (as their ~5-9% coverage predicted). Blood alone is slightly weaker than image v3 (0.50) overall but marginally better on the Cancer class. Restricting to rows with ≥1 core lab doesn't help (0.456), so missingness isn't the main limiter — blood just carries limited signal for this 3-way split. Artifacts + val predicted probabilities (for Phase 3 fusion) in `experiments/blood_baseline/`.
- ✅ **Text baseline**: `scripts/train_text_baseline.py` — frozen Bio_ClinicalBERT (mean-pooled) + logistic-regression head. Two settings because the labels are partly report-derived (Pneumonia 65% / Cancer 54% / Normal 100% depend on report text): **(A) INDICATION section only** (referring clinician's reason-for-exam, never used in labeling — the honest number) → macro F1 **0.51** (Normal 0.53, Pneumonia 0.69, **Cancer 0.32**); **(B) full report** → macro F1 0.68 but leakage-inflated (not a capability estimate). Leakage analysis: on the ICD-only val subset (report can't leak) A and B are ~equal (0.30 vs 0.33), so B's lead is entirely the report-derived subset. **Text is the best unimodal for the Cancer class** (0.32 vs image 0.17 / blood 0.20). Artifacts + both settings' val probabilities in `experiments/text_baseline/`.
- **Phase 2 takeaway**: image 0.50 / blood 0.46 / text 0.51 — all similar overall but with different per-class strengths (image→Pneumonia, text→Cancer), i.e. genuinely complementary. Fusion (Phase 3) has an empirical case.
- ⬜ Consider the cascade approach (Normal vs Abnormal, then Pneumonia vs Cancer) as an alternative framing to help the Cancer class.
- ✅ Cancer-class decision (post-fusion): fusion lifted Cancer F1 0.32 → 0.43 without any
  special handling. Further gains (focal loss / cascade / threshold tuning) are deferred —
  not worth the added complexity right now given the improvement; revisit only if final
  test-set numbers demand it.

## Phase 3 — Multimodal Fusion
✅ Done (late fusion) — fused val macro F1 **0.61** vs best single modality 0.51 (+0.10)
- **Late (decision-level) fusion**: stack the three Phase 2 baselines' 3-class probability
  vectors (9 features) → final label. `scripts/train_fusion.py`.
- OOF plumbing: `scripts/generate_oof_predictions.py` (blood + text, 5-fold subject-grouped
  StratifiedGroupKFold) and Kaggle kernel `medreason-ai-image-oof-val-predictions`
  (image, same folds) produce out-of-fold train predictions so the meta-model isn't fit
  on the base models' own training rows. Val predictions saved per modality too.
- Result (val): **logpool (geometric mean) 0.613**, mean-prob 0.600, unweighted logreg
  stack 0.585, balanced logreg 0.562, hgb stack 0.561. **Parameter-free pooling beats the
  trained stackers** (only 9 inputs, well-calibrated base probs → trained meta-models
  overfit / over-correct). Per class: Normal 0.53→0.62, Pneumonia 0.69→0.80,
  **Cancer 0.32→0.43** (biggest relative lift — the weak class benefits most).
- Ablation (mean rule): all-three (0.60) > best pair blood+text (0.576) > image+text
  (0.557) > image+blood (0.493). Text contributes most, image least.
- Leakage guard: text inputs are Setting A (INDICATION section only), never full-report.
- Artifacts in `experiments/fusion/`. **Embedding-level / attention fusion not needed** —
  late fusion already clears the bar comfortably.

## Phase 4 — Explainability
🔄 In progress — `explain/` package, one module per modality, each loads a saved model
and returns a common contract (predicted class, probabilities, ranked signed
attributions, one-line summary) + a matplotlib figure.
- **Blood** (`explain/blood.py`) — ✅ SHAP `TreeExplainer` (tree_path_dependent, exact) on
  the saved `HistGradientBoostingClassifier`. Attributions in log-odds; horizontal bar plot.
- **Text** (`explain/text.py`) — ✅ SHAP with a word-level (`\W+`) Text masker over the
  full embed→logreg pipeline; per-word effect on P(predicted). ~60–90 s/case on CPU
  (`max_evals=300`). Renders the indication with words shaded by effect.
- **Image** (`explain/image.py`) — ✅ Grad-CAM: keep image→backbone→18 pathology probs→our
  3-class logit differentiable, hook the backbone's last conv map, backprop the predicted
  logit (input needs `requires_grad` since the backbone is frozen). Plus pathology-channel
  attribution (`d(logit)/d(prob) · prob`) — names *which* of the 18 xrv channels drove it.
- `explain/run.py` — unified entrypoint: all three for one `study_id`, prints + saves
  PNGs + JSON. This is what the Phase 5 agent will call.
- Decoupled from training code (feature/preprocessing logic duplicated by hand, in sync).
- Local quirk fixed: `import torch` before sklearn/shap or torch's DLL init fails on Windows.
- Verified on one confident-correct case per class + two cross-modality-conflict cases
  (text/blood right, image wrong on Cancer) — explanations are clinically coherent.
- ⬜ deferred to UI/demo phases: a combined one-page per-case view; figure polish.
Core explainability capability is complete; the agent (Phase 5) can consume `explain/run.py`.

## Phase 5 — Agentic Layer
⬜ Not started
- Agent Orchestrator: tool-calling shell around the trained models (image/blood/text tools + fusion).
- Belief State: probability distribution over hypotheses, updated per tool call.
- Consistency Checker: cross-modality conflict detection (e.g. image says Cancer, blood/text say Normal).
- Missing Information Planner: decide whether more input is actually needed based on current confidence.
- Design rule to preserve: the agent orchestrates and reasons; the trained models diagnose.

## Phase 6 — Evidence Retrieval (RAG)
⬜ Not started
- Build a small corpus of relevant papers/guidelines for the 3 target diseases.
- Qdrant vector DB setup for retrieval.
- Evidence Planner: build an optimized query from prediction + explainability + belief state (not just the raw label).
- Local LLM via Ollama for final explanation generation — RAG explains, it does not diagnose.

## Phase 7 — Backend & API
⬜ Not started
- FastAPI service wrapping the full pipeline (upload → prediction → belief state → consistency check → explanation → evidence).
- PostgreSQL for case/result storage.
- Docker containerization of the full stack.

## Phase 8 — Integration, Testing, Documentation
⬜ Not started
- End-to-end testing across the full pipeline.
- Write up methodology and empirical findings (dataset linking rate, label validation approach, class imbalance handling, per-modality results) — this doubles as thesis material.
- Final polish pass.

### Two kinds of evaluation
- **Quantitative (thesis number):** one batch run of the full pipeline over all 2,099
  rows of `data/processed/test.csv` — the file already has `image_path`, `report_path`,
  the 17 blood cols, `final_label`. Script resolves the image, extracts the INDICATION
  section from the report (same `extract_indication` as training — leakage rule holds),
  runs image+blood+text+fusion+agent, compares to `final_label`. Report macro F1 /
  per-class / confusion / calibration **plus** agent-layer stats (how often the
  consistency checker fires, how often it escalates / requests more data). `test.csv`
  stays sealed until this run; develop Phases 4–7 against `val.csv`.
- **Interactive demo (viva):** the UI, one case at a time.

### Phase 8b — Demo preparation (for the final-year defense)
- **Case-curation script** — scan `val.csv` (never `test.csv`) for cases matching each
  edge-case profile, produce a vetted shortlist of ~15 study IDs with notes; freeze
  ~8–12 for the presentation into `demo/cases/` (gitignored — MIMIC images).
  Profiles to hit: clean Normal / Pneumonia / Cancer; **cross-modality conflict**
  (modalities disagree + consistency checker fires); **low confidence** (fused max prob
  ~0.4–0.55, triggers missing-info planner); **no blood labs provided** (graceful
  degradation); **model wrong but flags it** (low confidence / fired conflict — this is
  a strength to show, not hide); **non-CXR / garbage image** (needs a lightweight OOD
  input guard — build one).
- **UI:** "load example case" dropdown populating the form from `demo/cases/`.
- **Inputs a user provides:** (1) frontal chest X-ray PNG/JPG; (2) clinical indication /
  reason-for-exam as short free text — NOT a radiology report (leakage-safe + realistic:
  pre-read you have the film, labs, referral reason); (3) up to 17 blood labs, all
  optional, "no bloods" checkbox; (4) age/sex optional.
- **Framing for modest accuracy (fusion ≈ 0.61 macro F1):** this is a systems-engineering
  project (agentic pipeline, explainability, belief state, consistency checking, RAG,
  real backend), not a benchmark-accuracy project. Demo the *reasoning and explanations
  working*. A case the model gets wrong but the consistency checker catches → highlight.
- **De-risk before the defense:**
  - offline-first — all model weights vendored locally, zero runtime downloads
  - Ollama latency on 4 GB VRAM: small quantized model (Qwen2.5-3B / Phi-3), pre-warm,
    stream the RAG text after showing prediction+explainability, pre-cache explanations
    for the frozen demo cases
  - serving-path preprocessing must match torchxrayvision training exactly (grayscale,
    normalize ~[-1024,1024], center-crop, resize 224) — verify against known cases
  - rehearse Docker cold start (Postgres + Qdrant + Ollama + API); reset script to clear
    the case table between runs
  - record a full screen-capture of the demo end-to-end as a projector/laptop-failure backup
- **Data provenance:** get PhysioNet credentials before the defense (free, ~1–2 weeks,
  supervisor as reference) — closes the "is your data use authorized?" question.

---

## Where things are running
- **Data pipeline & exploration scripts**: run locally (`scripts/`, `scripts/exploration/`) — lightweight, no GPU needed.
- **Model training**: runs remotely on Kaggle Kernels (free GPU), not on the local laptop — see the Kaggle section in the main conversation / below for how to check on it.
- **Backend/agent/API (future phases)**: will run locally in VS Code, with Docker for Postgres/Qdrant and Ollama as a local service.
