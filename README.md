# MedReason-AI

**Agentic Multi-Modal Medical Diagnostic Assistant with Explainable AI and Evidence
Retrieval** — a final-year capstone.

Given a chest X-ray + the linked admission's blood labs + the radiology report, the system
predicts one of **3 classes — Normal / Pneumonia / Lung Cancer** — then (planned) an agent
layer reasons over the prediction: tracks a belief state, checks cross-modality
consistency, decides whether more information is needed, and retrieves supporting
literature (RAG) to *explain* the result — never to diagnose.

The goal of the project is learning to design a **production-grade AI system**, not
chasing benchmark accuracy. See [CLAUDE.md](CLAUDE.md) for the full project brief and
[ROADMAP.md](ROADMAP.md) for phase-by-phase status.

## Status

| Phase | State |
|-------|-------|
| 0 — Vision & scope | ✅ done |
| 1 — Data foundation | ✅ done — 15,453-sample joined dataset |
| 2 — Unimodal baselines | ✅ done — image 0.50 / blood 0.46 / text 0.51 (val macro F1) |
| 3 — Multimodal fusion | ⬜ next |
| 4 — Explainability | ⬜ |
| 5 — Agentic layer | ⬜ |
| 6 — RAG evidence retrieval | ⬜ |
| 7 — Backend & API (FastAPI + PostgreSQL + Qdrant + Docker) | ⬜ |
| 8 — Integration / testing / writeup | ⬜ |

## Data — not included in this repo

The dataset is built from **MIMIC-CXR** and **MIMIC-IV v3.1** (PhysioNet credentialed
data). Its license prohibits redistributing the data or identifier-carrying derivatives,
so **`data/`, the processed CSVs, embeddings, and per-study prediction files are
git-ignored.** To reproduce, obtain your own PhysioNet credentials, download the sources,
and run the pipeline below.

How the dataset is constructed (linking algorithm, label rule, blood-feature tiers) is
documented in [CLAUDE.md §3](CLAUDE.md).

## Layout

```
scripts/               data pipeline (run from repo root, local, no GPU)
  exploration/          feasibility probes, disease mapping, report labeler, label builder
  build_final_dataset.py / split_dataset.py
  train_blood_baseline.py    blood/tabular baseline (sklearn, local CPU)
  train_text_baseline.py     text baseline (frozen Bio_ClinicalBERT + logreg, local CPU)
kaggle_upload/         image-model training (runs on Kaggle free GPU, not locally)
experiments/           baseline results — RESULTS.md / metrics.json (weights & preds git-ignored)
data/                  git-ignored (see above)
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline (after placing MIMIC sources under `data/raw/`)

```bash
python scripts/exploration/check_linking_feasibility.py
python scripts/exploration/map_disease_labels.py
python scripts/exploration/save_resolved_links.py
python scripts/exploration/label_reports.py
python scripts/exploration/build_final_labels.py
python scripts/build_final_dataset.py
python scripts/split_dataset.py
# baselines
python scripts/train_blood_baseline.py
python scripts/train_text_baseline.py
```

Image-model training is driven via the `kaggle` CLI against
`kaggle_upload/baseline_kernel/` — see [CLAUDE.md §4](CLAUDE.md).
