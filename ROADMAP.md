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
- ⬜ Optional: revisit the still-weak Cancer class (focal loss / oversampling / threshold tuning) — but fusion may address it, so decide after Phase 3's first result.

## Phase 3 — Multimodal Fusion
⬜ Not started
- Combine image + blood + text embeddings into a single fusion model.
- Compare fused performance against each unimodal baseline — fusion should be justified empirically, not assumed.
- Decide fusion strategy (early/late/attention-based) based on what the unimodal results suggest.

## Phase 4 — Explainability
⬜ Not started
- Grad-CAM for the image model (localizes what the CNN attended to).
- SHAP for the blood/tabular model (feature attribution).
- Attention visualization for the text model.
- These attach to whichever models exist after Phase 3, so this phase follows fusion, not before.

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

---

## Where things are running
- **Data pipeline & exploration scripts**: run locally (`scripts/`, `scripts/exploration/`) — lightweight, no GPU needed.
- **Model training**: runs remotely on Kaggle Kernels (free GPU), not on the local laptop — see the Kaggle section in the main conversation / below for how to check on it.
- **Backend/agent/API (future phases)**: will run locally in VS Code, with Docker for Postgres/Qdrant and Ollama as a local service.
