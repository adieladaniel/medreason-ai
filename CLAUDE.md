# CLAUDE.md — MedReason-AI

Instructions and project context for Claude Code. Read this first every session, then
skim [ROADMAP.md](ROADMAP.md) for phase-by-phase status. Update both when state changes.

---

## 1. What this project is

**MedReason-AI** — a final-year capstone building an **Agentic Multi-Modal Medical
Diagnostic Assistant with Explainable AI and Evidence Retrieval** ("Version 1.5", a
foundation, not a clinical product).

Given a chest X-ray + linked blood labs + the radiology report, the system predicts one
of **3 classes — Normal / Pneumonia / Lung Cancer** — then an agent layer reasons over
the prediction: tracks a belief state, checks cross-modality consistency, decides if more
information is needed, and retrieves supporting literature (RAG) to *explain* (never to
diagnose).

### Framing that must guide every suggestion
The point of the project is **learning to design a production-grade AI system**, not
chasing benchmark accuracy. When helping, explain the *why* behind engineering choices
and favour production patterns (agents, explainability, RAG, evidence retrieval,
proper API/DB/containerisation). Every proposed feature must clear four bars:
**high learning value · production relevance · real engineering practice · reasonable
complexity.** If it doesn't, cut it.

### Hard scope boundaries
- **In scope:** image/blood/text analysis, multimodal fusion, disease prediction,
  explainability (Grad-CAM / SHAP / text attention), agentic orchestration, RAG evidence
  retrieval, FastAPI + PostgreSQL + Qdrant + Docker backend.
- **Out of scope (do not propose):** treatment/dosage/prescription, autonomous medical
  decisions, patient follow-up, multi-agent collaboration, clinical deployment.
- **Design rule:** the agent orchestrates and reasons; the trained models diagnose; RAG
  only explains. Never let the orchestrator emit a diagnosis directly or let RAG
  influence the predicted label.

### Constraints
- **₹0 development cost.** Everything runs locally or on free tiers. No paid APIs, no
  cloud GPU. LLM component runs locally via Ollama (Llama / Qwen / Mistral candidates).
- **Local machine:** RTX 3050 Laptop (4 GB VRAM), 15.7 GB RAM, Windows 11, PowerShell.
  Too small for model training — see §4.
- **Stack:** PyTorch / Transformers / scikit-learn, FastAPI, PostgreSQL, Qdrant, Docker,
  Git. Ollama for the local LLM.

---

## 2. Current status (2026-09-01)

| Phase | State | Notes |
|-------|-------|-------|
| 0 — Vision & scope | ✅ done | 3-class scope finalised empirically (TB/COVID dropped) |
| 1 — Data foundation | ✅ done | 15,453-sample joined dataset built + split |
| 2 — Unimodal baselines | ✅ done | image 0.50 / blood 0.46 / text 0.51 (val macro F1); complementary per-class |
| 3 — Multimodal fusion | ✅ done | late fusion, val macro F1 **0.61** (+0.10 over best single) |
| 4 — Explainability | ⬜ | **next** |
| 5 — Agentic layer | ⬜ | |
| 6 — RAG evidence retrieval | ⬜ | |
| 7 — Backend & API | ⬜ | |
| 8 — Integration / testing / writeup | ⬜ | |

### Image baseline results (val set, n=2,315)
- **v1** frozen ImageNet DenseNet121 + linear head, 8 ep → macro F1 **0.45** (Cancer F1 0.12)
- **v2** last block unfrozen, discriminative LR + cosine, 20 ep → macro F1 **0.46** (Cancer F1 0.14) — marginal; ImageNet features are the bottleneck, not undertraining
- **v3 (current best)** frozen `torchxrayvision` DenseNet121 pretrained on MIMIC-CXR
  (`densenet121-res224-mimic_ch`) as a feature extractor → its 18 pathology-probability
  outputs feed an MLP head (18→64→3), class-weighted loss, 15 ep, cosine, best epoch 7.
  → macro F1 **0.50** (Normal 0.56 / Pneumonia 0.77 / **Cancer 0.17, precision 0.14**).
  Code: [kaggle_upload/baseline_kernel/train_baseline.py](kaggle_upload/baseline_kernel/train_baseline.py).
  Lesson: *which* pretrained features mattered far more than *how much* fine-tuning.

### Blood/tabular baseline (val set) — done 2026-08-29
`scripts/train_blood_baseline.py`, runs locally on CPU. sklearn
`HistGradientBoostingClassifier` (native NaN handling) on the 17 lab columns + NLR/PLR
derived ratios, `class_weight="balanced"`. → macro F1 **0.46** (Normal 0.45 / Pneumonia
0.73 / **Cancer 0.20**). Logreg 0.40, majority-class floor 0.27. Top labs: hemoglobin,
albumin, LDH, WBC, lymphocytes. CRP/ESR contribute ~nothing (coverage too low).
Restricting to rows with ≥1 core lab → 0.456, so missingness isn't the bottleneck;
blood just carries limited signal for this split. Artifacts + val predicted probabilities
(for Phase 3 fusion) in `experiments/blood_baseline/`.

### Text baseline (val set) — done 2026-08-29
`scripts/train_text_baseline.py`, runs locally on CPU (~15 min first run to embed; caches
`emb_*.npy`). Frozen **Bio_ClinicalBERT** (mean-pooled) + logistic-regression head.
**Leakage matters here:** the labels were partly built from the reports (Pneumonia 65% /
Cancer 54% / Normal 100% of studies have report-derived label evidence — see §3), so a
full-report classifier partly reads its own answer. Two settings:
- **A — `INDICATION` section only (the honest number):** referring clinician's
  reason-for-exam, never used in labeling. → macro F1 **0.51** (Normal 0.53 / Pneumonia
  0.69 / **Cancer 0.32**).
- **B — full report:** macro F1 0.68 but **leakage-inflated — do not cite as text-modality
  performance.** On the ICD-only val subset (report can't leak) A and B are ~equal
  (0.30 vs 0.33), proving B's lead is the report-derived subset.
Artifacts + both settings' val probabilities in `experiments/text_baseline/`.

### Phase 2 conclusion
Image 0.50 / blood 0.46 / text 0.51 — similar overall, **different per-class strengths**
(image → Pneumonia F1 0.77; text → Cancer F1 0.32; blood → middle). The modalities are
genuinely complementary, so **Phase 3 fusion is empirically justified.** Each baseline
saved its val predicted probabilities keyed by `study_id` for fusion.

### Phase 3 — late multimodal fusion (val set) — done 2026-09-01
Stack the three baselines' 3-class probability vectors (9 features) → final label.
- **OOF plumbing:** `scripts/generate_oof_predictions.py` (blood + text, 5-fold
  subject-grouped `StratifiedGroupKFold`, seed 42) and Kaggle kernel
  `medreason-ai-image-oof-val-predictions` (image, code in
  `kaggle_upload/image_oof_kernel/`, same folds) produce out-of-fold train predictions so
  the meta-model isn't fit on the base models' own training rows. Per-modality val
  predictions saved too. Image kernel trick: extract the frozen backbone's 18-dim
  features once, then all fold training is on cached `[N,18]` matrices (~10 min total).
- **Fusion** (`scripts/train_fusion.py`, local): **logpool (geometric mean of the
  probability vectors) → val macro F1 0.613**, mean-prob 0.600, unweighted logreg stack
  0.585, class-balanced logreg 0.562, HGB stack 0.561. **Parameter-free pooling beats the
  trained stackers** — 9 inputs + calibrated base probs means a trained meta-model just
  overfits / over-corrects. Per class: Normal 0.53→0.62, Pneumonia 0.69→0.80,
  **Cancer 0.32→0.43**.
- Ablation (mean rule): all-three 0.60 > blood+text 0.576 > image+text 0.557 >
  image+blood 0.493. Text contributes most, image least.
- **Leakage guard:** text inputs are Setting A (INDICATION only), never full-report.
- Artifacts in `experiments/fusion/`. Embedding-level / attention fusion not pursued —
  late fusion already clears the bar.

**Cancer class:** fusion lifted it 0.32 → 0.43 with no special handling. Focal loss /
cascade / threshold tuning deferred — revisit only if the final test numbers require it.

### Phase 4 — explainability (in progress, 2026-09-02)
`explain/` package, one module per modality, each loads a saved model and returns a common
contract `{predicted_class, probabilities, attributions:[{feature,value,effect,direction}],
summary}` + a matplotlib figure. `explain/run.py` runs all three for a `study_id` and
saves PNGs + JSON — this is the interface the Phase 5 agent will call.
- **`explain/blood.py`** — SHAP `TreeExplainer` (exact, tree_path_dependent) on the HGB
  model; contributions in log-odds; horizontal bar figure.
- **`explain/text.py`** — SHAP with a word-level (`\W+`) Text masker over the whole
  embed→logreg pipeline; per-word effect on P(predicted class). ~60–90 s/case on CPU
  (`max_evals=300`) — fine for pre-computed demo cases, too slow for live UI (cache them).
- **`explain/image.py`** — Grad-CAM: image→backbone→18 pathology probs→our 3-class logit
  kept differentiable, hook the backbone's last conv map, backprop the predicted logit
  (**input must `requires_grad_(True)`** — backbone is frozen). Plus pathology-channel
  attribution `d(logit)/d(prob)·prob` naming which of the 18 xrv channels drove it.
  torchxrayvision weights pre-downloaded to `~/.torchxrayvision/models_data/` (its own
  download printer crashes on Windows cp1252 — fetch the `.pt` with curl).
- Runs local, CPU. `import torch` must precede sklearn/shap (gotcha 7).
- Rendered explanations of real cases go to `experiments/explain_samples/` — **gitignored**
  (X-ray images + indication text).

**The test set (`data/processed/test.csv`, 2,099 samples) has never been touched. Keep it
that way until final evaluation.**

---

## 3. Data — how the dataset was built

**Sources:** MIMIC-CXR (Kaggle mirror `nikeshreddypatlolla/mimic-cxr-dataset` — images +
reports + metadata) and MIMIC-IV v3.1 hospital module (`admissions`, `patients`,
`labevents`, `diagnoses_icd`, `d_labitems`). Raw data lives under `data/raw/` (large;
`labevents.csv` alone is ~18 GB — always query it with **DuckDB**, never load into pandas).

**Linking algorithm (do not re-derive — this was decided after evaluating alternatives):**
MIMIC-CXR has no `hadm_id`. Reconstruct it: `subject_id` → `study_datetime` → find the
admission where `admittime ≤ study_datetime ≤ dischtime` → assign that `hadm_id` → pull
that admission's labs + diagnoses. Tie-break multi-admission matches by nearest
`admittime`. Only ~48% of CXR studies link to a valid admission — this is the real
ceiling, plan for it.

**Sample granularity:** one sample = one CXR study + its report + the linked admission's
blood panel + diagnosis. Study-level, not patient-level.

**Labels (OR rule — code in
[scripts/exploration/build_final_labels.py](scripts/exploration/build_final_labels.py);
note its docstring says "AND" but the SQL is OR — the SQL is authoritative):**
- Pneumonia = ICD code **OR** positive report mention (checked first, so it wins ties)
- Lung Cancer = ICD code **OR** positive report mention
- Normal = report says "No Finding" **AND** admission has no Pneumonia/Cancer/**TB** ICD code
- anything else (other pathology) → `Excluded`, dropped
Report mentions come from a custom negation-aware labeler (spaCy + negspacy), not the
heavyweight CheXpert/NegBio labelers.

**Blood features (17 columns, tiered):**
- Tier 1 core — WBC, RBC, Hemoglobin, Hematocrit, Platelets (~94% coverage, reliable)
- Tier 1 differential — Neutrophils, Lymphocytes, Monocytes (~51%)
- Tier 2 — Albumin (46%), LDH (40%), **CRP (7.5%), ESR (3.7%)** — treat as sparse/optional,
  never required inputs
- Tier 3 — Sodium, Potassium, Glucose, Creatinine, Urea (~94%)
Value chosen per test = the measurement closest in time to `study_datetime`.

**Final dataset:** 15,453 samples — Pneumonia 10,165 / Normal 4,277 / Lung Cancer 1,011 —
across 4,803 subjects. **Subject-level** stratified 70/15/15 split (train 11,039 / val
2,315 / test 2,099); no patient crosses splits. Output: `data/processed/{final_dataset,
train,val,test}.csv`, columns: `subject_id, study_id, hadm_id, final_label, image_path,
report_path, <17 blood cols>, split`.

### Pipeline script order (all local, no GPU, `python scripts/...` from repo root)
1. `scripts/exploration/check_linking_feasibility.py` — feasibility probe
2. `scripts/exploration/map_disease_labels.py` — resolve multi-admission links + ICD→disease map
3. `scripts/exploration/save_resolved_links.py` → `data/interim/resolved_study_links.csv`
4. `scripts/exploration/label_reports.py` — negation-aware report labeler → `report_labels.csv`
5. `scripts/exploration/build_final_labels.py` — merge ICD + report signals → `final_study_labels.csv`
6. `scripts/build_final_dataset.py` — assemble image/report/blood table → `data/processed/final_dataset.csv`
7. `scripts/split_dataset.py` — subject-level split → `train/val/test.csv`

Support scripts: `check_blood_coverage.py`, `test_report_labeler.py`.
To change labeling strictness, edit params atop `build_final_labels.py` and rerun 5–7.

---

## 4. Model training — runs on Kaggle, not locally

All training runs on **Kaggle Kernels** (free GPU, user's account `adieladaniel42`),
driven entirely via the `kaggle` CLI (`kernels push/status/output`). The local laptop is
never used for training.

**Kaggle assets (all private):**
- `adieladaniel42/medreason-ai-processed` — the train/val/test/final CSVs
- `adieladaniel42/densenet121-imagenet-weights` — pretrained weights (uploaded manually)
- kernel `adieladaniel42/medreason-ai-baseline-image-classifier`, code in
  `kaggle_upload/baseline_kernel/` (Phase 2 image baseline v3)
- kernel `adieladaniel42/medreason-ai-image-oof-val-predictions`, code in
  `kaggle_upload/image_oof_kernel/` (Phase 3 image OOF + val predictions). NOTE: the
  Kaggle slug is derived from the *title*, so `id` in kernel-metadata.json must slugify to
  the same string — mismatched title/id creates a differently-named kernel.

**Infra gotchas already solved — reuse, don't rediscover:**
1. `kaggle` CLI ≥ 2.2.3 needs the token also at `~/.kaggle/access_token` (plain string)
   for write ops, not just `kaggle.json`.
2. GPU + internet on kernels requires account **phone verification** (done) — without it
   kernels silently fall back to CPU/no-internet.
3. Kaggle's preinstalled torch (cu128) dropped sm_60 (P100). Pin
   `torch==2.4.1+cu121` / `torchvision==0.19.1` via pip at script start — works on both
   P100 and T4. `--accelerator` / `machine_shape` flags don't reliably pick the GPU.
4. Dataset mount path is `/kaggle/input/datasets/<owner>/<slug>/...` — resolve
   dynamically with recursive glob, never hardcode.
5. A freshly created dataset needs a few minutes to reach `status: ready` before it can
   be attached.
6. Max 2 concurrent GPU sessions per account.

**Precedent for the remaining modalities:** prefer domain-pretrained encoders over
generic ones (e.g. ClinicalBERT/BioBERT for text, not vanilla BERT) — that was the lever
that worked for images.

7. Windows local quirk: importing `sklearn` before `torch` in the same process can break
   torch's DLL init (`WinError 1114`). In any local script that uses both, `import torch`
   first. (Kaggle/Linux unaffected.)

---

## 5. What to do next

**Phases 2 and 3 are done** (§2). Fusion (logpool of the 3 modalities' probabilities)
reaches val macro F1 0.61. Next:

1. **Phase 4 — explainability.** Attach per-modality explanations to the models that
   exist now:
   - **Image:** Grad-CAM on the torchxrayvision backbone. Subtlety — the Phase 2/3 head
     consumes the backbone's 18 pathology *probabilities*, not conv features, so Grad-CAM
     targets the backbone's own conv layers w.r.t. the relevant pathology logit(s) rather
     than "our" 3-class head. Decide: explain the pathology channels that drive our
     prediction, or add a thin conv-feature head to CAM against.
   - **Blood:** SHAP on the `HistGradientBoostingClassifier` (`shap.TreeExplainer`).
   - **Text:** attention / token-attribution over the INDICATION text — either
     attention rollout on Bio_ClinicalBERT or a model-agnostic method (e.g. LIME /
     integrated gradients) on the frozen-embedding + logreg pipeline.
   - Keep it a **separate `explain/` module** that loads the saved models — don't
     entangle it with training code.
2. **Then Phase 5** (agentic layer) orchestrates these + the fusion output.
3. Cancer-class polish (focal loss / cascade / thresholds) stays deferred — only if the
   sealed test set demands it at the end.

**Repo (done 2026-09-01):** private GitHub repo `adieladaniel/medreason-ai`, `main`
branch, `origin` remote. `.gitignore` excludes all of `data/`, model binaries
(`*.pth/*.joblib/*.npy`), and per-study prediction CSVs — **MIMIC is PhysioNet
credentialed data, never commit it, not even to a private repo.** `requirements.txt` +
`README.md` exist. Commit/push only when the user asks; branch off `main` for changes.
Still to do: the agreed `app/{api,agent,pipelines,services,database,config,models,utils}`
structure (create when Phase 7 backend work starts), `docs/`, `tests/`, `docker/`.

Work the iterative loop per component: **Design → Implement → Review → Test → Improve →
Proceed.** One component at a time, not big upfront builds.

---

## 6. Conventions

- **Scripts run from repo root**, paths are relative (`data/raw/...`). Keep that.
- **Query `labevents.csv` (and other large MIMIC tables) with DuckDB**, never pandas.
- Each script has a docstring explaining *why*, not just what — match that style.
- Training code targets Kaggle's environment (pinned torch install, dynamic path
  resolution); it is not expected to run locally.
- Keep the **test split sealed** until final evaluation.
- Any new labeling / dataset change → rerun the pipeline (§3), don't hand-edit CSVs.
- Update [ROADMAP.md](ROADMAP.md) status markers and this file's §2 when a phase moves.

---

## 7. Persistent memory

Claude Code keeps cross-session memory at
`C:\Users\HP\.claude\projects\c--Users-HP-Desktop-finalyearproject\memory\` — `MEMORY.md`
is its index. Those notes and this file should agree; if they drift, the code and
`ROADMAP.md` win, then update both.
