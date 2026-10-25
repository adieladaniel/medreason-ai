# MedReason-AI

An agentic multi-modal medical diagnostic assistant with explainable AI and evidence
retrieval.

The system takes a chest X-ray, the linked admission's blood labs, and the clinical
indication for the exam, and predicts one of three classes: Normal, Pneumonia, or Lung
Cancer. An agent layer then reasons over the prediction. It tracks a belief state, checks
whether the modalities agree, decides whether more information is needed, and retrieves
supporting literature (RAG) to explain the result rather than to diagnose. The trained models
diagnose, the agent orchestrates, and retrieval can never change the predicted label.

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
| 5. Agentic layer | core done (belief state, consistency checker, missing-information planner) |
| 6. RAG evidence retrieval | done (44-paper library, hybrid search with reranking, constrained writer, faithfulness check) |
| 7. Backend and API (FastAPI, PostgreSQL, Qdrant, Docker) | in progress, storage layer done |
| 8. Integration, testing, writeup | not started |

The test split (2,099 studies) has not been used. All numbers above are validation numbers.
Each phase is described in [ROADMAP.md](ROADMAP.md), and the Phase 7 design and its reasons are in
[docs/PHASE7_DESIGN.md](docs/PHASE7_DESIGN.md).

## Progress dashboard

A local web UI that shows everything done so far: phase tracker, data pipeline
statistics, per-model results and training curves, fusion comparisons (ROC, precision
recall, calibration, confusion matrices), the explainability gallery, the evidence
retrieval results, and a reliability page (confidence intervals, calibration, abstention).

```bash
python -m uvicorn app.main:app --port 8000
# then open http://localhost:8000
```

It reads the real result files under `experiments/` and rebuilds its data file
(`app/data/dashboard.json`) on startup when a source is newer, so new results show up
without editing the UI. The gallery needs `experiments/explain_samples/`, which is
git-ignored.

## Case API

The same server runs a case through the pipeline in the background. Start Ollama first if you want the written
note (`ollama serve`), then:

```bash
python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000/#/analyze for the Analyze page: upload a chest X-ray, type an indication, enter any
labs, and watch the prediction, the explanations and the cited evidence fill in as each stage finishes. The
interactive API documentation is at http://localhost:8000/docs, where `POST /api/cases` can be tried with a file
upload. It takes a chest X-ray (PNG or JPEG), a clinical indication and lab values as a JSON object, any of them
optional but at least one, and answers at once with a case id. `GET /api/cases/{id}` shows whatever stage results
exist so far (the prediction arrives first, then explanations, then the cited evidence), and
`GET /api/cases/{id}/stream` follows it live. Cases are deleted after 30 days, or at once with
`DELETE /api/cases/{id}`. There is no login: this is a local demonstration and must not be given real patient
data.

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
app/                   FastAPI app, progress dashboard, agent, evidence retrieval, storage
  agent/                orchestrator, belief state, consistency checker, missing-information planner
  rag/                  corpus builder, hybrid retrieval, evidence planner, writer, faithfulness check, evaluations
  database/             Phase 7 storage: tables, sessions, repository
  config.py             settings from MEDREASON_* environment variables
tests/                 189 tests (plus 26 slow ones: browser and Postgres), no models or services needed (python -m pytest)
docs/                  Phase 7 design, machine and environment notes
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

Build the evidence library and its index, then run the tests:

```bash
python -m app.rag.build_corpus      # downloads the open-access papers listed in app/rag/corpus_sources.json
python -m app.rag.index --build     # embeds them into the local Qdrant folder
python -m pytest                    # about 15 seconds
```

Run one case through the agent and the evidence chain (the text explanation step takes one to two
minutes on CPU, and the writer needs Ollama running with `llama3.2`):

```bash
python -m app.agent.orchestrator --study-id 53383543
python -m app.rag.explain_case --study-id 53383543
```
