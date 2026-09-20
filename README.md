# MedReason-AI

An agentic multi-modal medical diagnostic assistant with explainable AI and evidence
retrieval. Final-year capstone project.

The system takes a chest X-ray, the linked admission's blood labs, and the clinical
indication for the exam, and predicts one of three classes: Normal, Pneumonia, or Lung
Cancer. A planned agent layer then reasons over the prediction. It tracks a belief state,
checks whether the modalities agree, decides whether more information is needed, and
retrieves supporting literature (RAG) to explain the result rather than to diagnose.

The aim of the project is to learn how to build a production-grade AI system. Benchmark
accuracy is secondary. See [CLAUDE.md](CLAUDE.md) for the full project brief and
[ROADMAP.md](ROADMAP.md) for phase-by-phase status.

## Status

| Phase | State |
|-------|-------|
| 0. Vision and scope | done |
| 1. Data foundation | done, 15,453-sample joined dataset |
| 2. Unimodal baselines | done, image 0.50 / blood 0.46 / text 0.51 (val macro F1) |
| 3. Multimodal fusion | done, val macro F1 0.61 |
| 4. Explainability | core done (SHAP + Grad-CAM per modality) |
| 5. Agentic layer | not started |
| 6. RAG evidence retrieval | not started |
| 7. Backend and API (FastAPI, PostgreSQL, Qdrant, Docker) | not started |
| 8. Integration, testing, writeup | not started |

## Progress dashboard

A local web UI that shows everything done so far: phase tracker, data pipeline
statistics, per-model results and training curves, fusion comparisons (ROC, precision
recall, calibration, confusion matrices), and the explainability gallery.

```bash
python -m uvicorn app.main:app --port 8000
# then open http://localhost:8000
```

It reads the real result files under `experiments/` and rebuilds its data file
(`app/data/dashboard.json`) on startup when a source is newer, so new results show up
without editing the UI. The gallery needs `experiments/explain_samples/`, which is
git-ignored.

## Data is not included in this repo

The dataset is built from MIMIC-CXR and MIMIC-IV v3.1 (PhysioNet credentialed data). The
license does not permit redistributing the data or identifier-carrying derivatives, so
`data/`, the processed CSVs, cached embeddings, and per-study prediction files are all
git-ignored. To reproduce the work, obtain your own PhysioNet credentials, download the
sources, and run the pipeline below.

How the dataset is built (the linking algorithm, the label rule, the blood-feature tiers)
is documented in [CLAUDE.md](CLAUDE.md) section 3.

## Layout

```
scripts/               data pipeline, run from repo root, local, no GPU
  exploration/          feasibility probes, disease mapping, report labeler, label builder
  build_final_dataset.py / split_dataset.py
  train_blood_baseline.py    blood/tabular baseline (sklearn, local CPU)
  train_text_baseline.py     text baseline (frozen Bio_ClinicalBERT + logreg, local CPU)
  generate_oof_predictions.py / train_fusion.py    Phase 3 fusion
explain/               Phase 4 per-modality explainers (SHAP, Grad-CAM)
app/                   FastAPI app and progress dashboard (first slice of the Phase 7 backend)
kaggle_upload/         image-model training, runs on Kaggle free GPU, not locally
experiments/           baseline results (RESULTS.md, metrics.json). Weights and per-study preds git-ignored
data/                  git-ignored (see above)
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline (after placing the MIMIC sources under `data/raw/`)

```bash
python scripts/exploration/check_linking_feasibility.py
python scripts/exploration/map_disease_labels.py
python scripts/exploration/save_resolved_links.py
python scripts/exploration/label_reports.py
python scripts/exploration/build_final_labels.py
python scripts/build_final_dataset.py
python scripts/split_dataset.py

python scripts/train_blood_baseline.py
python scripts/train_text_baseline.py
python scripts/generate_oof_predictions.py
python scripts/train_fusion.py                       # needs image OOF preds from Kaggle
```

Image-model training is driven through the `kaggle` CLI against
`kaggle_upload/baseline_kernel/`. See [CLAUDE.md](CLAUDE.md) section 4.
