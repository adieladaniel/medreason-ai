# CLAUDE.md (MedReason-AI)

Instructions and project context for Claude Code. Read this first every session, then
skim [ROADMAP.md](ROADMAP.md) for phase-by-phase status. Update both when state changes.

Writing style for this file and all project docs: plain prose, no em dashes, no
AI-writing tics. See section 6.

---

## 1. What this project is

**MedReason-AI** is a final-year capstone building an agentic multi-modal medical
diagnostic assistant with explainable AI and evidence retrieval. It is called "Version
1.5": a foundation for later work, not a clinical product.

Given a chest X-ray, the linked blood labs, and the clinical indication for the exam, the
system predicts one of 3 classes (Normal, Pneumonia, Lung Cancer). An agent layer then
reasons over the prediction. It tracks a belief state, checks whether the modalities
agree, decides whether more information is needed, and retrieves supporting literature
(RAG) to explain the result. It never uses RAG to diagnose.

### Framing that must guide every suggestion

The point of the project is learning to design a production-grade AI system, not chasing
benchmark accuracy. When helping, explain the reasoning behind engineering choices and
prefer production patterns (agents, explainability, RAG, evidence retrieval, proper
API/DB/containerisation). Every proposed feature must clear four bars: high learning
value, production relevance, real engineering practice, reasonable complexity. If it does
not, cut it.

### Hard scope boundaries

- In scope: image/blood/text analysis, multimodal fusion, disease prediction,
  explainability (Grad-CAM, SHAP, text attribution), agentic orchestration, RAG evidence
  retrieval, FastAPI + PostgreSQL + Qdrant + Docker backend.
- Out of scope (do not propose): treatment, dosage, or prescription; autonomous medical
  decisions; patient follow-up; multi-agent collaboration; clinical deployment.
- Design rule: the agent orchestrates and reasons, the trained models diagnose, RAG only
  explains. Never let the orchestrator emit a diagnosis directly, and never let RAG
  influence the predicted label.

### Constraints

- Zero development cost. Everything runs locally or on free tiers. No paid APIs, no cloud
  GPU. The LLM component runs locally via Ollama (Llama, Qwen, or Mistral candidates).
- Local machine: RTX 3050 Laptop (4 GB VRAM), 15.7 GB RAM, Windows 11, PowerShell. Too
  small for model training, see section 4.
- Stack: PyTorch, Transformers, scikit-learn, FastAPI, PostgreSQL, Qdrant, Docker, Git.
  Ollama for the local LLM.

---

## 2. Current status (2026-09-02)

| Phase | State | Notes |
|-------|-------|-------|
| 0. Vision and scope | done | 3-class scope finalised empirically (TB and COVID dropped) |
| 1. Data foundation | done | 15,453-sample joined dataset built and split |
| 2. Unimodal baselines | done | image 0.50 / blood 0.46 / text 0.51 (val macro F1), complementary per class |
| 3. Multimodal fusion | done | late fusion, val macro F1 0.61 (+0.10 over best single) |
| 4. Explainability | core done | SHAP + Grad-CAM per modality, `explain/` package |
| 5. Agentic layer | not started | next |
| 6. RAG evidence retrieval | not started | |
| 7. Backend and API | not started | |
| 8. Integration, testing, writeup | not started | |

### Image baseline results (val set, n=2,315)

- v1: frozen ImageNet DenseNet121 + linear head, 8 epochs, macro F1 0.45 (Cancer F1 0.12).
- v2: last block unfrozen, discriminative LR + cosine, 20 epochs, macro F1 0.46 (Cancer F1
  0.14). Marginal gain. ImageNet features are the bottleneck, not undertraining.
- v3 (current best): frozen `torchxrayvision` DenseNet121 pretrained on MIMIC-CXR
  (`densenet121-res224-mimic_ch`) as a feature extractor. Its 18 pathology-probability
  outputs feed an MLP head (18 to 64 to 3), class-weighted loss, 15 epochs, cosine, best
  epoch 7. Macro F1 0.50 (Normal 0.56 / Pneumonia 0.77 / Cancer 0.17, precision 0.14).
  Code: [kaggle_upload/baseline_kernel/train_baseline.py](kaggle_upload/baseline_kernel/train_baseline.py).
  Lesson: which pretrained features you use mattered far more than how much fine-tuning.

### Blood/tabular baseline (val set), done 2026-08-29

`scripts/train_blood_baseline.py`, runs locally on CPU. sklearn
`HistGradientBoostingClassifier` (native NaN handling) on the 17 lab columns plus NLR and
PLR derived ratios, `class_weight="balanced"`. Macro F1 0.46 (Normal 0.45 / Pneumonia
0.73 / Cancer 0.20). Logreg 0.40, majority-class floor 0.27. Top labs: hemoglobin,
albumin, LDH, WBC, lymphocytes. CRP and ESR contribute almost nothing (coverage too low).
Restricting to rows with at least one core lab gives 0.456, so missingness is not the
bottleneck. Blood just carries limited signal for this split. Artifacts and val predicted
probabilities (for Phase 3 fusion) in `experiments/blood_baseline/`.

### Text baseline (val set), done 2026-08-29

`scripts/train_text_baseline.py`, runs locally on CPU (about 15 min first run to embed,
then caches `emb_*.npy`). Frozen Bio_ClinicalBERT (mean-pooled) + logistic-regression
head.

Leakage matters here. The labels were partly built from the reports (Pneumonia 65% /
Cancer 54% / Normal 100% of studies have report-derived label evidence, see section 3),
so a full-report classifier partly reads its own answer. Two settings:

- A, `INDICATION` section only (the honest number): the referring clinician's
  reason-for-exam, never used in labeling. Macro F1 0.51 (Normal 0.53 / Pneumonia 0.69 /
  Cancer 0.32).
- B, full report: macro F1 0.68 but leakage-inflated. Do not cite it as text-modality
  performance. On the ICD-only val subset (where the report cannot leak) A and B are about
  equal (0.30 vs 0.33), which shows B's lead comes from the report-derived subset.

Artifacts and both settings' val probabilities in `experiments/text_baseline/`.

### Phase 2 conclusion

Image 0.50 / blood 0.46 / text 0.51. Similar overall, different per-class strengths
(image is strongest on Pneumonia F1 0.77, text on Cancer F1 0.32, blood in the middle).
The modalities are genuinely complementary, so Phase 3 fusion is empirically justified.
Each baseline saved its val predicted probabilities keyed by `study_id` for fusion.

### Phase 3, late multimodal fusion (val set), done 2026-09-01

Stack the three baselines' 3-class probability vectors (9 features) into a final label.

- OOF plumbing: `scripts/generate_oof_predictions.py` (blood and text, 5-fold
  subject-grouped `StratifiedGroupKFold`, seed 42) and the Kaggle kernel
  `medreason-ai-image-oof-val-predictions` (image, code in `kaggle_upload/image_oof_kernel/`,
  same folds) produce out-of-fold train predictions so the meta-model is not fit on the
  base models' own training rows. Per-modality val predictions are saved too. The image
  kernel trick: extract the frozen backbone's 18-dim features once, then all fold training
  runs on cached `[N,18]` matrices (about 10 min total).
- Fusion (`scripts/train_fusion.py`, local): logpool (geometric mean of the probability
  vectors) reaches val macro F1 0.613. Mean-prob 0.600, unweighted logreg stack 0.585,
  class-balanced logreg 0.562, HGB stack 0.561. Parameter-free pooling beats the trained
  stackers, because 9 inputs plus calibrated base probabilities means a trained meta-model
  just overfits or over-corrects. Per class: Normal 0.53 to 0.62, Pneumonia 0.69 to 0.80,
  Cancer 0.32 to 0.43.
- Ablation (mean rule): all three (0.60) beat blood+text (0.576), which beats image+text
  (0.557), which beats image+blood (0.493). Text contributes most, image least.
- Leakage guard: text inputs are Setting A (INDICATION only), never full-report.
- Artifacts in `experiments/fusion/`.

Feature-level fusion was tried as a documented ablation, done 2026-09-20
(`scripts/train_fusion_mlp.py`): concatenate the frozen per-modality features (image's 18
pathology probabilities, blood's 19 features imputed with missingness flags, text's
768-dim embedding reduced to 32 by PCA chosen on a dev split) and train one MLP, instead
of stacking predicted probabilities. No OOF needed, since these are frozen encoder
outputs, not model predictions fit on our labels. Train is split fit/dev 85/15
(subject-grouped, stratified), dev used only for early stopping and the PCA size, val
touched once. Result: val macro F1 0.586, below late fusion's 0.613, and the same
architecture scored 0.655 on dev, so the gap is overfitting (88 features, only 774 cancer
rows in train). Same modality pattern as late fusion (all three beats every pair, text
most, image least). Late fusion remains the one used downstream. Artifacts in
`experiments/fusion_mlp/`.

Cancer class: fusion lifted it from 0.32 to 0.43 with no special handling. Focal loss,
cascade, and threshold tuning are deferred. Revisit only if the final test numbers
require it.

### Phase 4, explainability (core done, 2026-09-02)

`explain/` package, one module per modality. Each loads a saved model and returns a common
contract: `{predicted_class, probabilities, attributions:[{feature,value,effect,direction}],
summary}`, plus a matplotlib figure. `explain/run.py` runs all three for a `study_id` and
saves PNGs plus JSON. That is the interface the Phase 5 agent will call.

- `explain/blood.py`: SHAP `TreeExplainer` (exact, tree_path_dependent) on the HGB model.
  Contributions in log-odds. Horizontal bar figure.
- `explain/text.py`: SHAP with a word-level (`\W+`) Text masker over the whole
  embed-then-logreg pipeline. Per-word effect on P(predicted class). About 60 to 90
  seconds per case on CPU (`max_evals=300`). Fine for pre-computed demo cases, too slow
  for a live UI, so cache them.
- `explain/image.py`: Grad-CAM. Keep image, backbone, 18 pathology probs, and the 3-class
  logit as one differentiable chain, hook the backbone's last conv map, backprop the
  predicted logit. The input tensor must call `requires_grad_(True)` because the backbone
  is frozen. Also a pathology-channel attribution, `d(logit)/d(prob) * prob`, naming which
  of the 18 xrv channels drove the prediction. torchxrayvision weights are pre-downloaded
  to `~/.torchxrayvision/models_data/`, because its own download printer crashes on
  Windows cp1252 (fetch the `.pt` with curl).
- Runs local, CPU. `import torch` must come before sklearn or shap (gotcha 7).
- Rendered explanations of real cases go to `experiments/explain_samples/`, which is
  git-ignored (X-ray images and indication text).

### Progress dashboard (2026-09-19)

`app/` is the first slice of the Phase 7 backend: a FastAPI app plus a static single-page
UI (Chart.js vendored under `app/static/vendor/`, no Node, no CDN) showing everything done
so far. Pages: overview and phase tracker, data pipeline stats, model results and the
Kaggle training curve, fusion (method comparison, ablation, confusion matrices, ROC,
precision-recall, calibration, modality agreement), explainer gallery, roadmap. Run
`python -m uvicorn app.main:app --port 8000` from the repo root and open
http://localhost:8000.

- `app/dashboard_data.py` builds `app/data/dashboard.json` from the real result files
  (`experiments/*/metrics.json`, the val prediction CSVs, the Kaggle training log, the
  pipeline logs in `scripts/`, and phase status parsed from `ROADMAP.md`). `app.main`
  rebuilds it on startup when a source is newer, so new experiments appear without
  editing the UI. When a phase adds results, add a section to the builder and a page or
  card in `app/static/app.js`.
- The JSON holds aggregates only (no study, subject, or admission IDs) and is committed.
  The gallery reads `experiments/explain_samples/`, which is git-ignored.
- The only hand-entered numbers are image baselines v1 and v2 (`IMAGE_VERSIONS` in the
  builder), because those runs kept no logs.
- Verified by driving headless Edge: all six pages render with no console errors and the
  class tabs and confusion-matrix selector work.

The test set (`data/processed/test.csv`, 2,099 samples) has never been touched. Keep it
that way until final evaluation.

---

## 3. Data, how the dataset was built

Sources: MIMIC-CXR (Kaggle mirror `nikeshreddypatlolla/mimic-cxr-dataset`, with images,
reports, and metadata) and MIMIC-IV v3.1 hospital module (`admissions`, `patients`,
`labevents`, `diagnoses_icd`, `d_labitems`). Raw data lives under `data/raw/`. It is
large: `labevents.csv` alone is about 18 GB, so always query it with DuckDB, never load
it into pandas.

Linking algorithm (do not re-derive, this was decided after evaluating alternatives).
MIMIC-CXR has no `hadm_id`. Reconstruct it: take `subject_id` and `study_datetime`, find
the admission where `admittime <= study_datetime <= dischtime`, assign that `hadm_id`, and
pull that admission's labs and diagnoses. Tie-break multi-admission matches by nearest
`admittime`. Only about 48% of CXR studies link to a valid admission. That is the real
ceiling, so plan for it.

Sample granularity: one sample is one CXR study, its report, and the linked admission's
blood panel and diagnosis. Study-level, not patient-level.

Labels (OR rule, code in
[scripts/exploration/build_final_labels.py](scripts/exploration/build_final_labels.py)).
Its docstring says "AND" but the SQL is OR, and the SQL is authoritative.

- Pneumonia = ICD code OR positive report mention (checked first, so it wins ties)
- Lung Cancer = ICD code OR positive report mention
- Normal = report says "No Finding" AND the admission has no Pneumonia, Cancer, or TB ICD code
- anything else (other pathology) becomes `Excluded` and is dropped

Report mentions come from a custom negation-aware labeler (spaCy + negspacy), not the
heavyweight CheXpert or NegBio labelers.

Blood features (17 columns, tiered):

- Tier 1 core: WBC, RBC, Hemoglobin, Hematocrit, Platelets (about 94% coverage, reliable)
- Tier 1 differential: Neutrophils, Lymphocytes, Monocytes (about 51%)
- Tier 2: Albumin (46%), LDH (40%), CRP (7.5%), ESR (3.7%). Treat as sparse and optional,
  never required inputs.
- Tier 3: Sodium, Potassium, Glucose, Creatinine, Urea (about 94%)

The value chosen per test is the measurement closest in time to `study_datetime`.

Final dataset: 15,453 samples (Pneumonia 10,165 / Normal 4,277 / Lung Cancer 1,011) across
4,803 subjects. Subject-level stratified 70/15/15 split (train 11,039 / val 2,315 / test
2,099), no patient crosses splits. Output: `data/processed/{final_dataset,train,val,test}.csv`,
columns: `subject_id, study_id, hadm_id, final_label, image_path, report_path, <17 blood
cols>, split`.

### Pipeline script order (all local, no GPU, run `python scripts/...` from repo root)

1. `scripts/exploration/check_linking_feasibility.py`: feasibility probe
2. `scripts/exploration/map_disease_labels.py`: resolve multi-admission links and ICD-to-disease map
3. `scripts/exploration/save_resolved_links.py` writes `data/interim/resolved_study_links.csv`
4. `scripts/exploration/label_reports.py`: negation-aware report labeler, writes `report_labels.csv`
5. `scripts/exploration/build_final_labels.py`: merge ICD and report signals, writes `final_study_labels.csv`
6. `scripts/build_final_dataset.py`: assemble the image/report/blood table, writes `data/processed/final_dataset.csv`
7. `scripts/split_dataset.py`: subject-level split, writes `train/val/test.csv`

Support scripts: `check_blood_coverage.py`, `test_report_labeler.py`. To change labeling
strictness, edit the params at the top of `build_final_labels.py` and rerun steps 5 to 7.

---

## 4. Model training, runs on Kaggle not locally

All training runs on Kaggle Kernels (free GPU, user account `adieladaniel42`), driven
through the `kaggle` CLI (`kernels push/status/output`). The local laptop is never used
for training.

Kaggle assets (all private):

- `adieladaniel42/medreason-ai-processed`: the train/val/test/final CSVs
- `adieladaniel42/densenet121-imagenet-weights`: pretrained weights (uploaded manually)
- kernel `adieladaniel42/medreason-ai-baseline-image-classifier`, code in
  `kaggle_upload/baseline_kernel/` (Phase 2 image baseline v3)
- kernel `adieladaniel42/medreason-ai-image-oof-val-predictions`, code in
  `kaggle_upload/image_oof_kernel/` (Phase 3 image OOF and val predictions). Note: the
  Kaggle slug is derived from the title, so `id` in kernel-metadata.json must slugify to
  the same string, or you get a differently-named kernel.

Infra gotchas already solved, reuse them, do not rediscover:

1. `kaggle` CLI 2.2.3 or newer needs the token also at `~/.kaggle/access_token` (plain
   string) for write ops, not just `kaggle.json`.
2. GPU and internet on kernels require account phone verification (done). Without it,
   kernels silently fall back to CPU with no internet.
3. Kaggle's preinstalled torch (cu128) dropped sm_60 (P100). Pin `torch==2.4.1+cu121` and
   `torchvision==0.19.1` via pip at script start. That works on both P100 and T4. The
   `--accelerator` and `machine_shape` flags do not reliably pick the GPU.
4. The dataset mount path is `/kaggle/input/datasets/<owner>/<slug>/...`. Resolve it
   dynamically with a recursive glob, never hardcode.
5. A freshly created dataset needs a few minutes to reach `status: ready` before it can be
   attached.
6. Maximum 2 concurrent GPU sessions per account.
7. Windows local quirk: importing `sklearn` before `torch` in the same process can break
   torch's DLL init (`WinError 1114`). In any local script that uses both, `import torch`
   first. Kaggle and Linux are not affected.

Precedent for the remaining modalities: prefer domain-pretrained encoders over generic
ones (for example ClinicalBERT or BioBERT for text, not vanilla BERT). That was the lever
that worked for images.

---

## 5. What to do next

Phases 2, 3, and 4 core are done (section 2). Fusion (logpool of the 3 modalities'
probabilities) reaches val macro F1 0.61. The three explainers work and are verified.

Next:

1. Phase 5, agentic layer. An orchestrator that calls the model tools plus
   `explain/run.py`, maintains a belief state (probability distribution over the 3
   classes, updated per tool call), runs a consistency checker (flags cross-modality
   conflicts, for example image says Cancer but blood and text say Normal), and a
   missing-information planner (decides whether more input is worth requesting given
   current confidence). Keep the design rule: the agent orchestrates and reasons, the
   models diagnose.
2. Then Phase 6 (RAG evidence retrieval) and Phase 7 (backend and API).
3. Cancer-class polish (focal loss, cascade, thresholds) stays deferred. Only if the
   sealed test set demands it at the end.

Repo (done 2026-09-01): private GitHub repo `adieladaniel/medreason-ai`, `main` branch,
`origin` remote. `.gitignore` excludes all of `data/`, model binaries
(`*.pth`, `*.joblib`, `*.npy`), per-study prediction CSVs, and `experiments/explain_samples/`.
MIMIC is PhysioNet credentialed data, so never commit it, not even to a private repo.
`requirements.txt` and `README.md` exist. Commit and push only when the user asks, and
branch off `main` for changes. `app/` now exists with the dashboard (`main.py`,
`dashboard_data.py`, `static/`, `data/`). Still to do: the agreed subfolders
`app/{api,agent,pipelines,services,database,config,models,utils}` as Phases 5 to 7 land,
plus `docs/`, `tests/`, `docker/`. The Analyze page (upload a case, agent steps,
evidence) is planned as an addition to this same app.

Work the iterative loop per component: design, implement, review, test, improve, proceed.
One component at a time, not big upfront builds.

---

## 6. Conventions

- Scripts run from repo root, paths are relative (`data/raw/...`). Keep that.
- Query `labevents.csv` and other large MIMIC tables with DuckDB, never pandas.
- Each script has a docstring explaining why, not just what. Match that style.
- Training code targets Kaggle's environment (pinned torch install, dynamic path
  resolution). It is not expected to run locally.
- Keep the test split sealed until final evaluation.
- Any new labeling or dataset change means rerunning the pipeline (section 3), not
  hand-editing CSVs.
- Update [ROADMAP.md](ROADMAP.md) status markers and this file's section 2 when a phase
  moves.
- Git commits: plain message, no `Co-Authored-By` trailer and no AI attribution of any
  kind. The user's git identity is the sole author.
- Prose in this file, ROADMAP.md, README files, `*.txt`, docstrings, and the eventual
  writeup: no em dashes, no en dashes as punctuation, and avoid the patterns commonly
  flagged as AI writing (puffery, forced rule of three, "it is worth noting", opening
  with "Moreover" or "Importantly", heavy bold and nested bullets). Plain sentences.

---

## 7. Persistent memory

Claude Code keeps cross-session memory at
`C:\Users\HP\.claude\projects\c--Users-HP-Desktop-finalyearproject\memory\`. `MEMORY.md`
is its index. Those notes and this file should agree. If they drift, the code and
`ROADMAP.md` win, then update both.
