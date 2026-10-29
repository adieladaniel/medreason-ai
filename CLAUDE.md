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

## 2. Current status (2026-09-20)

| Phase | State | Notes |
|-------|-------|-------|
| 0. Vision and scope | done | 3-class scope finalised empirically (TB and COVID dropped) |
| 1. Data foundation | done | 15,453-sample joined dataset built and split |
| 2. Unimodal baselines | done | image 0.50 / blood 0.46 / text 0.51 (val macro F1), complementary per class |
| 3. Multimodal fusion | done | late fusion, val macro F1 0.61 (+0.10 over best single) |
| 4. Explainability | core done | SHAP + Grad-CAM per modality, `explain/` package |
| 5. Agentic layer | core done | orchestrator, belief state, consistency checker, planner in `app/agent/` |
| 6. RAG evidence retrieval | done | corpus, retrieval, planner, constrained writer, faithfulness check, topic gate and dashboard page done and run on real cases |
| 7. Backend and API | done | storage, staged pipeline, case API, Analyze page, observability, containers and the end-to-end run |
| 8. Integration, testing, writeup | done | sealed test run once (macro F1 0.609), non-X-ray guard, other-pathology measurement, demo cases, results write-up |

### Image baseline results (val set, n=2,315)

- v1: frozen ImageNet DenseNet121 + linear head, 8 epochs, macro F1 0.45 (Cancer F1 0.12).
- v2: last block unfrozen, discriminative LR + cosine, 20 epochs, macro F1 0.46 (Cancer F1
  0.14). Marginal gain. ImageNet features are the bottleneck, not undertraining.
- v3 (current best): frozen `torchxrayvision` DenseNet121 pretrained on MIMIC-CXR
  (`densenet121-res224-mimic_ch`) as a feature extractor. Its 18 pathology-probability
  outputs (only 11 are trained for these weights, see below) feed an MLP head (18 to 64 to 3), class-weighted loss, 15 epochs, cosine, best
  epoch 7. Macro F1 0.50 (Normal 0.56 / Pneumonia 0.77 / Cancer 0.17, precision 0.14).
  Code: [kaggle_upload/baseline_kernel/train_baseline.py](kaggle_upload/baseline_kernel/train_baseline.py).
  Lesson: which pretrained features you use mattered far more than how much fine-tuning.
  The 11 trained outputs are Atelectasis, Consolidation, Pneumothorax, Edema, Effusion,
  Pneumonia, Cardiomegaly, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum. The
  other 7 slots (Infiltration, Emphysema, Fibrosis, Pleural_Thickening, Nodule, Mass, Hernia)
  have empty names and return a constant 0.5, which the library describes as untrained
  targets, so the head effectively has 11 informative inputs plus a bias. Nodule and Mass, the
  most cancer-relevant channels, are among the untrained ones; only Lung Lesion remains. Open
  question, unverified: the backbone was trained on MIMIC-CXR and the library does not say
  which split, so some val and test subjects may have been in its training images. Note it as a
  limitation in the writeup. The `all` weights train every output but their training data is not stated.

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
  embed-then-logreg pipeline. Per-word effect on P(predicted class). `max_evals=300`.
  Measured 2026-10-25 on a quiet machine: 0.4 to 0.9 s for a typical indication (the median
  is 8 words) and 10 to 17 s for the longest ones (95 to 153 words). An earlier note here said
  60 to 90 s; that was not reproduced, and probably came from a loaded machine or an unusual
  text. SHAP cannot run on fewer than two words: it crashed on "cough" and on an empty
  indication (about 4 percent of the validation set), so those now get an exact single-word
  effect, or none.
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
- The fusion page also shows the feature-fusion MLP ablation (`_fusion_mlp()` in the
  builder), added 2026-09-20 alongside that experiment.

### Phase 5, agentic layer (core done, 2026-09-20)

`app/agent/` package: `tools.py`, `belief.py`, `consistency.py`, `planner.py`,
`orchestrator.py`. Design rule enforced throughout: the orchestrator calls tools and
reasons about their outputs, it never produces a class prediction of its own.

- Tools are thin wrappers around `explain.image/blood/text`, which already return
  `{predicted_class, probabilities, attributions, summary}`. No new prediction logic.
- Belief state pools whichever modalities have run with the exact Phase 3 rule (geometric
  mean, `scripts/train_fusion.py` logpool). Recomputed from the full evidence set on every
  update, not folded in as a running product, so the belief after any subset of tools
  matches that subset's row in the Phase 3 ablation table exactly.
- Consistency checker's confidence language ("cases like this are correct N% of the
  time") reads real numbers from `app/data/dashboard.json` (`predictions.agreement`),
  falling back to the same numbers as constants if that file is missing.
- Missing-information planner's confidence thresholds (0.85 high, 0.55 low) are a design
  choice; which modality it recommends next is read from the real Phase 3 ablation
  ranking (text contributes most, then blood, then image), via
  `tools.MODALITY_PRIORITY`.
- CLI mirrors `explain/run.py`: `python -m app.agent.orchestrator --study-id ID
  [--drop image|blood|text]`. `--drop` simulates a missing modality, which is also how
  the Phase 8b demo cases (missing labs, cross-modality conflict) get produced.
- Verified against real validation cases: full agreement, a missing modality, a genuine
  conflict (text and blood correctly call Lung Cancer, image confidently calls Normal;
  the checker flags it and the pooled belief still lands on Cancer), and a
  single-modality low-confidence case that correctly recommends text next.
- Not yet wired to an API or the UI. The dashboard's "Analyze a case" link is still a
  placeholder pointing at this; that wiring is Phase 7 work.

### Phase 6, RAG evidence retrieval (in progress, 2026-10-11)

`app/rag/` package. Corpus step done: `app/rag/corpus_sources.json` lists 14 open-access
PubMed Central papers (pneumonia imaging and diagnosis, lung cancer imaging, lung cancer
lab markers, the pneumonia versus cancer differential). `python -m app.rag.build_corpus`
downloads each as JATS XML from Europe PMC, drops citation markers and boilerplate
sections, and writes `data/corpus/<PMCID>.json` with the text split by section plus
citation metadata (authors, journal, year, DOI, license). About 40,800 words in total.

- Licenses were checked per paper through Europe PMC's `license` field. Allowed: CC BY,
  CC BY-NC, CC BY-NC-SA, CC0. CC BY-NC-ND is excluded, since chunking is a derivative use.
- `data/` is git-ignored, so the corpus text is rebuilt from the manifest, not committed.
- Web access works from this environment: `curl` and the WebSearch and WebFetch tools
  both reach the internet. Earlier notes saying otherwise were wrong.
- Ollama CLI is installed (0.34.2) but no server is running and no models are confirmed
  pulled. Docker and a Postgres driver are not installed.

Retrieval core, stage 1 (2026-10-11, 14 papers, `chunker.py`, `index.py`, `eval_retrieval.py`,
`eval_chunking.py`, results `RESULTS.md` and `CHUNKING.md` in `experiments/rag_retrieval/`):
embedded Qdrant (a folder under `data/qdrant/`, one line to swap for a server in Phase 7),
sentence-packed passages of about 300 words with about 15 percent overlap embedded as "title.
heading. passage", and a comparison of five chunking strategies and four embedding models.
Semantic chunking did not beat plain sentence packing. Stage 1 used 19 queries, so it could
not separate good from great, and its eval scripts now give different numbers against the
larger corpus. Treat them as the record of how the first choices were made.

Retrieval improvements, stage 2 (2026-10-12): driven by a bigger evaluation, because stage 1
could not show improvements.

- Corpus: 27 open-access papers, 82,500 words, 474 passages. The 13 added papers were chosen
  from the models' own top features (hemoglobin, albumin, LDH, WBC, urea, platelets, sodium,
  lymphocytes) and image channels (edema, effusion, atelectasis), so coverage follows what the
  models actually use. All licenses checked, CC BY-NC-ND still excluded.
- Evaluation (`eval_queries.py`, `eval_pipeline.py`, `audit_proxy.py`): 54 queries (questions
  and the terse keyword form the evidence planner will emit) split into dev and test by topic
  group, plus 16 off-corpus queries (several deliberately near the corpus: tuberculosis, COPD,
  pulmonary embolism, pneumothorax). A passage is relevant if its topic is allowed and it
  contains the question's key terms. Variants are chosen on dev and confirmed on test.
  Random ranking scores p@1 0.05. `AUDIT.md` records a hand reading of 18 top results: the
  proxy never called an irrelevant passage relevant but passes weak ones, and it undercounts
  about half of the passages it rejects. Read levels as approximate, comparisons as sturdier.
- Result (`PIPELINE.md`), right passage at rank 1, dev / test: semantic only 0.73 / 0.57,
  keyword BM25 only 0.81 / 0.79, hybrid 0.96 / 0.75, hybrid plus MedCPT reranker 1.00 / 0.86.
  Keyword search beat the semantic model that stage 1 chose, partly because the queries share
  vocabulary with the papers, so this favors keyword search more than a real user would.
- Adopted pipeline (`retrieve.py`): hybrid search (dense plus BM25 merged by reciprocal rank
  fusion), then `ncbi/MedCPT-Cross-Encoder` reranks the top 5, then passages scoring below
  `MIN_RERANK_SCORE = 4.0` are dropped. An empty result means "no supporting evidence".
- Why MedCPT and not the smaller MiniLM reranker: both rank about equally (differences of one
  or two queries), but MedCPT's score abstains far better, with ROC AUC 0.997 against 0.953
  for telling answerable from off-corpus queries. With a threshold chosen on dev, it refused
  8 of 8 off-corpus test queries and kept 26 of 28 answerable ones (the two refused: a
  white-blood-cell query the corpus cannot answer, and a pulmonary-edema query at 3.94).
- Not adopted: query expansion (`hybrid.expand_query`). It did not help dense search, hybrid
  search, or reranking on dev or test. My expectation that it would fix the "elevated WBC"
  failure was wrong.
- Scoring detail that mattered: reranker scores must be raw logits. Sigmoid probabilities
  saturate at exactly 1.0, create ties that the sort breaks arbitrarily (it inflated the first
  MedCPT numbers), and make any threshold hair-thin.
- Latency, measured 2026-10-25 (`eval_latency.py`; the 54 answerable and 16 off-corpus queries, one at
  a time, CPU-only torch with 10 threads on 16 logical CPUs, AC power, Balanced plan). The same code gave
  three very different results depending on what else was running:
  quiet machine (Chrome, WhatsApp, Docker and WSL closed, VS Code open, 7.8 GB RAM free, `latency.json`,
  108 timings): lookup p50 1.1 s, p95 1.3 s, max 1.4 s, flat across the whole run. Chrome and VS Code open
  (run with 162 timings, raw samples overwritten, summary kept here): p50 3.9 s, p95 4.7 s, max 5.0 s.
  Chrome used heavily with 3.3 GB RAM free (`latency_loaded_run.json`, 54 timings, run from the user's own
  terminal): p50 16.4 s, p95 42.5 s, max 45 s, recovering to 7 to 8 s late in the run. So a lookup is about
  1 s when the laptop is quiet and several times slower when it is not. Memory pressure is the likely
  cause (free RAM was 3.3 GB in the slow run), but that was not isolated. It is not the torch thread
  count (4, 6, 8, 10 and 16 threads all gave p50 3.4 to 4.5 s in the Chrome-open state), and it is not
  foreground versus background (the user's foreground run was the slowest).
  Hybrid search alone is 0.05 to 0.1 s, so the cross-encoder is almost all of the time (about 0.2 s per
  passage when quiet). A first query after the process starts took 15 to 24 s (retriever build 1 to 6 s,
  then loading the reranker), measured with the OS file cache warm, so a start after a reboot will be
  slower. Say "about 1 s on a quiet laptop, 4 s with a browser open" and never one bare number. Close the
  browser before timing or demoing, and remember that Docker's WSL VM can hold up to 8 GB of the 16 GB.
  Depth 5, 10, and 20 gave the same quality. Fewer passages clear the threshold at depth 5 (often 1 to 3),
  which is the trade-off.
- Known limits: the reranker can rank a near-miss first with a high score (for "platelet to
  lymphocyte ratio in pneumonia" it put an NLR passage ahead of the paper on exactly that),
  and a high score means on-topic, not that the passage supports a claim. The explanation
  writer must quote what a passage states and say when none addresses a point. Provisional
  numbers: 16 off-corpus queries and 54 answerable ones, written by the author with the
  assistant.
Explanation chain, built 2026-10-11 (`planner.py`, `evidence.py`, `writer.py`,
`faithfulness.py`, `explain_case.py`, check set in `eval_faithfulness.py`):

- Planner: reads the agent result and writes up to 5 keyword queries, one per real driver. A
  lab becomes a query only if SHAP says it supported the prediction and its value is outside an
  approximate adult reference range ("pneumonia with low albumin"). Image channels at 0.5 or
  higher (one query per channel: a combined "edema and lung opacity" query scored 3.3 on the
  reranker where "lung opacity" alone scored 10.8), clinically meaningful words from the
  indication, and one differential query when modalities disagree and one of them predicted the
  other disease (put first so the cap never cuts it). Each query also carries the terms a
  sentence must mention to count as being about that finding. A modality whose own
  prediction differs from the final one never seeds driver queries, because its attributions
  explain a different class. A Normal prediction asks for nothing.
- Evidence: retrieves 2 passages per query, at most 5 in total, numbers them, records which
  query fetched each. A query with no passage above the threshold is reported as "no
  passage", never dropped silently.
- Writer: the first version gave llama3.2 (3B) the whole case and a rulebook. It cited
  passages that did not exist, wrote "associated with [2]" with nothing after it, and
  misreported which modalities agreed. So the design changed: sentences about the case
  (prediction, probabilities, agreement, drivers) are built by code from the agent's facts, and
  the model only writes one background sentence per retrieved passage, from that passage alone.
  With no passages the model is never called, so "no evidence" cannot turn into an invented
  citation. The predicted class in the output is copied from the agent. Topic gate: the writer
  model answered NOT_ADDRESSED in none of 164 tries, so relevance needed its own test. A
  sentence must mention the finding its query was written for ("opacity", "albumin"), else the
  passage is treated as off topic. This removed a faithful but misleading case report (persistent
  consolidation later found to be cancer, retrieved for a pneumonia case) and a CT-timing note.
  A driver with no kept sentence is listed as "no passage". The gate is conservative.
- Faithfulness check, verify then show: each LLM sentence must pass an NLI entailment test
  against the one passage it cites (DeBERTa-v3 MNLI/FEVER/ANLI, best 3-sentence window,
  threshold 0.30), a test that every number appears in the passage, and a wording test (no
  treatment or advice words, no "causes" or "proves"). Failed sentences are removed and counted.
- Checker validation (`eval_faithfulness.py`, 35 hand-made sentences against 5 real passages):
  every unfaithful sentence scored 0.19 or lower and every faithful one 0.48 or higher, so it
  separated the set at thresholds from 0.3 to 0.95 (the higher the threshold the more faithful
  sentences it wrongly drops). This includes the subtle case the numeric test cannot see: a
  real writer sentence that attached pneumonia mortality figures to the neutrophil-lymphocyte
  ratio, every number present in the passage, entailment 0.19. Limits: small set, written by
  the author, so the error types are the ones the author thought of. It is a sanity check, not
  a measured error rate on real output.
- On the four saved real cases: 3 of 4 background sentences passed on the pneumonia case (the
  NLR one was dropped), the cancer-versus-pneumonia conflict case got a correct passage on how
  mucinous adenocarcinoma mimics pneumonia, and two cases found no passage and said so. Image
  drivers (edema, opacity) found no passage in the library in both cases, a corpus gap.
- Known limits: a passage retrieved for one driver can be about another (a WBC query returned
  lymphocyte-count papers), so background sentences are shown as literature context, not as
  proof of a specific finding. A warm run takes 4 to 21 s; the first run after start took
  about 2.5 min (model loads), so Phase 7 must load models once and run this off the request path.
- Writer model comparison (`eval_writer.py`, `writer_eval.json`, 41 passages retrieved for the 24
  terse evaluation queries, checker as judge): llama3.2 3B passed 73% of its sentences, Qwen2.5
  3B passed 63%. Qwen's lower entailment (0.56 against 0.69) is the gap. Both write about 28
  words and take 3 to 4 s per item (about 2 s of that was a connection delay, see Phase 7 step 5, the same for
  both models). llama3.2 stays. Limits: 41 items, so the gap is about four
  sentences, and the judge is the checker, validated on 35 hand-made sentences, not on these.
- Parent-child expansion tested and not adopted: showing the writer the next 150 words of the
  same paper, while still checking against the cited passage, lowered the pass rate for both
  models (llama3.2 73% to 66%, Qwen 63% to 60%). The differences are a few sentences, so the
  result is "no evidence it helps", and the simpler setup stays.
- Of the 41 sentences llama3.2 wrote, 11 failed the check. Most were entailment failures that
  were real (a survival figure attached to the wrong marker, an anemia claim the passage did not
  make). One was a checker bug: the wording test treated "all-cause mortality" as a causal claim
  because a hyphen is a regex word boundary. Fixed. Treatment words inside a paper's own
  description still drop a faithful sentence, by design.
- Checker cost: scoring every 3-sentence window of a 300-word passage took about 10 s per
  sentence on CPU; keeping the 5 windows that share the most words with the sentence takes about
  6 s and gave the same scores on the 35-sentence set. The local torch is a CPU-only build
  (2.11.0+cpu), so the GPU is unused for this. A request needs a CUDA build or a smaller or
  quantised NLI model, which is Phase 7 work.
- Checker comparison (2026-10-12, `eval_checker.py`, `minicheck.py`, `checker_eval.json`): the
  current DeBERTa NLI checker against MiniCheck (EMNLP 2024) in its DeBERTa-v3-Large and
  Flan-T5-Large variants, reimplemented from the reference code because the package would not
  build here. Two labelled sets, each sentence against one passage: the 35 hand-made sentences,
  and the 41 real llama3.2 sentences, hand-labelled by reading them against the passage before
  any checker's score was looked at (36 faithful, 3 unfaithful, 2 borderline left out). Each
  checker used its own default threshold (0.30 for the NLI, 0.5 for MiniCheck), untuned.
  Faithful real sentences wrongly dropped: NLI 7 of 36, MiniCheck DeBERTa 3, MiniCheck Flan-T5 2.
  Unfaithful real sentences caught: NLI 2 of 3, both MiniCheck variants 0 of 3. Hand-made
  unfaithful caught: 19, 15 and 14 of 19 (the NLI threshold was tuned on that set, so its 19 is
  optimistic). Seconds per sentence on CPU: 8.0, 4.6 and 2.3. The MiniCheck models are faster and
  drop fewer good sentences but let all three subtle errors through, and the current checker
  catches more at the cost of dropping about one faithful sentence in five. A false accept
  misleads while a false reject only shortens the note, so the NLI checker stays.
- What none of them catch: the three unfaithful real sentences each blend true facts into a false
  statement (two findings fused into one, a group reversed, a general survival figure attached
  to a subgroup). Raising any checker's threshold until it catches all three drops 12 to 16 of
  the 36 faithful sentences, and Flan-T5 scored those three at 0.94, 0.94 and 0.82 against a
  faithful median of 0.96. About 8 percent of this writer's sentences were that kind, so the
  interface must present background as what a cited paper says, with the source one click
  away, never as a verified fact about this patient. Limits: one rater, three real negatives,
  so every rate on the unfaithful side rests on three sentences. A larger labelled set is the
  way to settle it, not another threshold.
- Download note: a 3 GB Hugging Face model stalled mid-download twice here. `curl -C -` resumed
  it (verify the SHA-256, which is the blob name in the cache, because a retry restarts at the
  original offset). `pip install` of the minicheck package failed to build, so do not rely on it.
- Corpus growth, stage 3 (2026-10-14, `eval_gaps.py`, `gap_probe_*.json`): instead of guessing
  gaps, every query the planner can emit (each blood phrase and image channel, both diseases, 52
  in all) was run through retrieval and counted as a gap when no returned passage mentions the
  finding. With 27 papers, 29 of 52 were gaps, far more than the edema case I had in mind. 17
  open-access papers were added from Europe PMC searches aimed at them (license checked, the
  builder rejects others): pneumonia with hemoglobin, platelets, urea and CURB-65, LDH, NLR and
  PLR, neutrophils and white blood cell count; lung cancer with white blood cell count,
  lymphocytes, CRP and inflammation indices; one lung imaging review and one lobar collapse case.
  The library is now 44 papers, 144,541 words, 811 passages. Gaps fell to 15 of 52, with none
  that had been covered becoming empty.
- Edema was mostly a phrasing problem: "pneumonia chest X-ray with pulmonary edema" scored 1.1
  because no passage is about both, while "pulmonary edema chest X-ray" scored 8.1 on a passage
  the library already held. The planner now gives each image query a fallback, "<finding> versus
  <disease> chest radiograph" (5.1 on the new imaging review), used only when the primary query
  returns nothing. With broader opacity terms (air-space, ground-glass) this brought gaps to 13
  of 52. The 13 left are mostly queries where little supportive literature should exist
  (leukopenia, hematocrit, cardiomegaly or pneumothorax with these diseases), so "no passage"
  is the right output there. Lung Cancer with lung opacity is still empty.
- Regression check on the original 54 queries (dev / test split unchanged), adopted pipeline,
  27 papers then 44: right passage first 1.00 / 0.857 then 1.00 / 0.893, MRR 0.944 then 0.968,
  abstention AUC 0.997 both, off-corpus queries refused 16 of 16 both, answerable queries kept
  52 then 53 of 54. The one-query change on test is within noise, so read it as "no harm".
  The weaker first-stage variants dipped a little with more documents (keyword only test
  0.786 to 0.714, plain hybrid 0.750 to 0.714), which the reranker absorbs. The 54 queries
  pre-date the new papers, so this checks for damage and does not prove the gain; the gain is
  the gap count. Baseline numbers are kept in `pipeline_27papers.json`.
- Effect on the real cases: white blood cell count now retrieves passages for the pneumonia
  case, and the 3B writer then wrote sentences with figures the passages do not support (for
  example a 15,000 per microlitre cut-off); the checker dropped both at entailment about 0.1, so
  only an albumin sentence survived. More coverage raises the chance the writer has something to
  summarise, and the checker decides what reaches the output. Writer quality is now the
  limiting factor for the share of drafted sentences that survive.
- Dashboard: an Evidence page (`evidencePage()` in `app/static/app.js`, `_rag()` in the builder)
  shows retrieval results, the checker's separation plot and the writer comparison.
- Left in Phase 6: nothing blocking. A faster checker (GPU or a smaller model) is tracked for
  Phase 7. Cardiomegaly and pneumothorax coverage was left empty on purpose.

### Reliability of the validation numbers (2026-10-12)

`scripts/eval_uncertainty.py` (local, about 30 s, validation only, writes
`experiments/uncertainty/metrics.json`, shown on the dashboard's Reliability page). It
reproduces the saved fusion score (0.6133) before doing anything else.

- Confidence intervals: cluster bootstrap over subjects (2,000 draws), because one patient has
  several studies. Fusion macro F1 0.613 (0.569 to 0.650). Fusion minus the best single model
  (text): +0.104 (+0.073 to +0.132), so the fusion gain is not sampling noise. Lung Cancer F1
  0.425 (0.302 to 0.524) and ROC AUC 0.855 (0.798 to 0.900). The image model's Lung Cancer AUC is
  0.527 (0.440 to 0.600), so on that class it cannot be told from guessing.
- Calibration: the pooled probabilities are under-confident. Temperature scaling (T = 0.53,
  fitted on out-of-fold train predictions only) cuts expected calibration error from 0.144 to
  0.028 and Brier from 0.432 to 0.398, and never changes the predicted class.
- Selective prediction: declining cases below confidence 0.55 (chosen on train to keep 80
  percent) kept 81 percent of val, and accuracy on answered cases went from 72 to 76 percent,
  macro F1 from 0.61 to 0.66. The cost falls unevenly: 32 percent of true Lung Cancer cases are
  declined against 20 percent of Normal and 17 percent of Pneumonia.
- Conformal sets (class-conditional, calibrated on out-of-fold train): coverage on val lands
  near target for every class, including Lung Cancer (80 percent target: 79 / 81 / 84, 90
  percent target: 88 / 90 / 91). The sets are large: 56 percent single-class at 80 percent
  coverage, 25 percent at 90 percent. Useful as "not sure between these", not as a sharper answer.
- Subgroups (sex, age band, recorded race): intervals overlap everywhere there are enough cases,
  so no gap is clearly more than noise, and the data is too thin to show none exists. Groups with
  fewer than 10 Lung Cancer cases (Hispanic, Asian, other race) are marked thin. Lung Cancer
  recall is lower for men (0.41 against 0.55, 44 against 73 cases) and for age 80 and over
  (0.18, 11 cases); both are suggestive only.
- Limits: out-of-fold train predictions come from fold models and val predictions from the
  full-train models, so calibration transfers approximately (val coverage shows how well).
  Conformal guarantees assume exchangeability and patients repeat. The test set is untouched.
  These checks were repeated once, unchanged, on the test split on 2026-10-29 (Phase 8 below): the calibration, the prediction-set coverage and
  the answered-only accuracy all held on unseen patients.

### Does the text input carry the project? (2026-10-25, `scripts/eval_dominance.py`, validation only)

Raised by testing the Analyze page: the text model decided most cases, and on the lung cancer case only the text model
found the cancer. Measured, with 95 percent intervals that resample patients:

- Text is the strongest input but not the only one that matters. Adding an input to the other two raises macro F1 by
  +0.105 for text (+0.064 to +0.145), +0.051 for blood (+0.021 to +0.078) and +0.026 for image (+0.011 to +0.041). All
  three intervals stay above zero, but the image adds the least, and its gain is small.
- The trained text model is close to a keyword rule. A crude rule written from clinical common sense and not tuned on
  these cases (a cancer word means Lung Cancer, an infection word means Pneumonia, otherwise Normal) scores macro F1
  0.502 against the text model's 0.509. So the text input mostly carries the referring clinician's own suspicion in
  words ("evaluate for pneumonia", "questionable metastasis"). It is a legitimate input, available before the film is
  read, but it is clinical context, not the AI understanding a report, and the project should be described that way.
- Text dominates the rare class, not the common ones. Per-class F1 (Normal / Pneumonia / Lung Cancer): image 0.56 / 0.73
  / 0.15, blood 0.45 / 0.73 / 0.20, text 0.53 / 0.69 / 0.32. For Pneumonia text is the weakest single model, and for
  Normal the image is slightly better. Only for Lung Cancer is text far ahead.
- Where the indication has no cancer or infection word (57 percent of cases, 1,320 of 2,315) text alone is the weakest
  (macro F1 0.43, accuracy 53 percent), image plus blood score 0.51 (accuracy 70 percent), and all three 0.56. That is
  where the image and blood models earn their place.
- The image model cannot see cancer here. Its Lung Cancer ROC AUC is 0.53 (earlier note). Likely reasons, not proven:
  the frozen backbone's Nodule and Mass channels are the untrained ones, the label is built from admission-level codes
  and reports so the film may show nothing, and there are only 1,011 cancer cases in total.
- Of the 117 validation cancers, text alone finds 71 (61 percent), image 34, blood 26. Averaging the three finds only
  58 (50 percent): it lost 18 cancers the text model found and gained 5. Eight cancers are found only by image and eight
  only by blood, and 28 (24 percent) by none. So "highest probability wins" is a poor rule for the rare class. It raises
  macro F1 (cancer F1 0.32 to 0.42) by cutting false alarms, at the cost of recall.
- Only 7 percent of cancer cases have "metastasis" in the indication, and text finds 88 percent of those. 60 percent have
  some cancer-type word.
- A cancer alert (flag "possible lung cancer" when the fused cancer probability reaches t, with t chosen on training
  out-of-fold predictions and applied once to validation) trades recall for false alarms: t = 0.358 catches 56 percent
  at 8.9 alerts per 100 cases (32 percent real), 0.304 catches 66 percent at 13.0 (26 percent real), 0.244 catches 76
  percent at 21.0 (18 percent real), 0.188 catches 84 percent at 33.0 (13 percent real). Cancer is 5.1 percent of cases.
  The conflict case seen on the page had a fused cancer probability of 0.22, so only the last setting flags it. This is
  a design option and one run; if adopted it needs a decision on how many false alarms are acceptable, wording that
  reports the model's signal without advising anything, and a single confirmation on the test split at the end.
- Limits: the keyword lists and the "no clue" subset are mine and unchecked by anyone else, 117 cancer cases make every
  cancer number wide, and these are validation numbers.
- What to say about the project honestly: the engineering (staged pipeline, explanations, evidence with verified
  sentences, calibration and abstention measured) is the substance. The model quality is modest, the text input is
  mostly clinician context, and Lung Cancer is hard. Done on 2026-10-26: prediction sets, a low-confidence headline and the cancer alert
  (see the Phase 7 notes). The image-weights experiment was then run and came out negative (2026-10-29,
  `experiments/image_backbones/RESULTS.md`): NIH and PadChest weights, which have trained Nodule and Mass channels and no MIMIC
  pretraining, did not meet the pre-registered adoption rule, and the current image model stays.

### Phase 7, backend and API (done 2026-10-28, started 2026-10-14)

Design and reasons: `docs/PHASE7_DESIGN.md`. Built one step at a time with a report after each.

- Step 1, storage (done): `app/config.py` (settings from `MEDREASON_*` environment variables or
  `.env`), `app/database/models.py` (cases, events, llm_calls; JSON is JSONB on Postgres),
  `session.py` (engine per URL, SQLite gets foreign keys and write-ahead logging, sessions are
  short and never shared between threads), `repository.py` (the only code that touches the
  tables; `jsonable` converts sets and numpy values, because pipeline output cannot go into a
  JSON column raw). Each stage result is saved in its own column as soon as it exists.
  Unfinished cases are marked failed at startup because the in-process worker does not survive a
  restart. `purge_older_than` implements the 30-day retention. 12 tests; breaking the
  in-place JSON edit and the restart recovery on purpose failed the right tests.
- Step 2, staged pipeline and job runner (done 2026-10-25): `app/pipelines/staged.py` (the stages),
  `app/pipelines/runner.py` (`run_job`, the `JobRunner` worker thread, and a CLI that runs one validation
  case through it), `app/agent/reasoning.py` (the one belief, consistency and plan function that both
  `run_case` and the staged prediction call). Stages: predicting, explaining (blood and image saved while
  text runs), retrieving, writing, verifying. Each result is saved the moment it exists, with a started and
  finished event and its seconds, and every model call is stored with its prompt. A failure after the first
  stage leaves the case "partial" with everything before it; a failed first stage is "failed". Explainer
  probabilities are compared with the stored prediction, and a difference is recorded, never applied. The
  retrieval lookup cache (`retrieve.py`) makes a repeated query free. `predict_image`, `predict_blood` and
  `predict_text` give probabilities without the attribution step.
  Measured on real validation cases (scratch database, quiet machine): the staged belief, consistency and
  plan are identical to `run_case`'s, and all three explainers matched the stored prediction exactly (largest
  difference 0.0). Seconds per stage: cold start (first case after launch) predicting 6.3, explaining 7.6,
  retrieving 19.3, writing 31.5, verifying 24.4, total 89; warm, same case again, 0.3, 1.0, 0.0 (lookups
  cached), 15.2, 20.3, total 37; a conflict case with one new lookup, 0.2, 0.9, 1.0, 12.9, 3.3, total 18. So
  the answer is ready in under a second when warm, and nearly all the waiting is the language model (about 3
  s per passage, of which about 2 s was a connection delay found and fixed in step 5) and the entailment
  checker. Staging matters for those stages, not for SHAP, which turned out to be quick.
- Two findings from that run. The writer is not reproducible even at temperature 0: the same case wrote a
  different albumin sentence the second time, and the checker passed the first (entailment 0.57) and dropped
  the second (0.01), so the shown text can differ between runs of the same case. And the text explainer
  crashed on one-word or empty indications; fixed (see the text explainer bullet), with tests.
- Step 3, the case API (done 2026-10-25): `app/api/{validation,views,cases}.py`, and `app/main.py` is now a
  `create_app(settings, runner)` factory (the module still exposes `app`, so `uvicorn app.main:app` is
  unchanged). Endpoints: `POST /api/cases` (multipart: image, indication, labs as a JSON object) answers 202 with
  the case id; `GET /api/cases`, `GET /api/cases/{id}`, `/events`, `/stream` (server-sent events: `stage`, `case`,
  `done`), `/llm-calls`, `/figures/{blood|image|text}.png`, `/image`, and `DELETE /api/cases/{id}`. Interactive
  docs at `/docs`. At startup it creates the tables, fails any case a dead process left unfinished, deletes cases
  past `retention_days` with their files, and starts the worker (`MEDREASON_WARM_UP_MODELS=true` loads every
  model first, about 2 GB). Input rules, all enforced before anything is stored: image PNG or JPEG decided from
  the bytes, at most 10 MB (413), at most 50 million pixels checked before decoding, each side at least 64, fully
  decoded once so a truncated file is refused; labs only the 17 known names, numbers from 0 to 1,000,000, null
  means not measured; indication at most 2000 characters, refused not cut; blank text or text with no letters or
  digits counts as absent; at least one usable input (422 naming the field); more than `max_queue` (10) cases
  waiting gives 429 with Retry-After. Server file paths never appear in a response, and `/image` only serves a
  file inside the upload folder. No login, deliberately (local demo, no real patients). Not checked: that the
  image is a chest X-ray, so any valid PNG or JPEG gets a prediction (the guard is planned for Phase 8b).
  Tested with a fake worker (32 API tests, 23 input tests; six deliberate breakages were each caught, and the
  first round of that check was invalid because of my own script and was redone; it also exposed that no test
  covered serving a path outside the upload folder, now added). Run for real: the server on port 8001 with the
  real models and a scratch database, a validation case sent over HTTP with its X-ray, and the live stream
  followed: prediction stored 0.5 s after the case started, explanations readable at 2 s, whole case done in 60
  s with the model warm-up already finished (explaining 5.9 s, retrieving 11.7, writing 18.8, verifying 21.1),
  the figures and the uploaded image served back byte for byte, a bad lab name refused before any model ran, and
  delete returned 204 then 404.
- Step 4, the Analyze page (done 2026-10-25): `app/static/analyze.js` (new), plus small changes to `app.js`
  (route arguments such as `#/analyze/<id>`, a hook that stops the live stream when the user leaves the page,
  shared helpers exposed as `window.MedReason`, and the page works even if `dashboard.json` is missing),
  `style.css` and `index.html`. The form is built from a new `GET /api/input-spec` (lab names, units and limits come
  from the same constants the server checks, so they cannot drift). Units were read from MIMIC itself and match
  the training medians: K/uL, g/dL, mg/L, mEq/L, mg/dL, IU/L, mm/hr, % (urea is urea nitrogen, the three
  differential counts are percentages of white cells). Submitting follows the case on the event stream with a
  polling fallback and draws each section when its stage has stored something: the stage tracker, the
  prediction, whether the inputs agree with a one-step-at-a-time table, per-input cards (uploaded X-ray,
  Grad-CAM, SHAP bars, word shading, top contributions), the cited evidence note with each reference's passage one
  click away and a "not proof about this patient" caveat, an audit table, links to the raw case and every model
  prompt, delete, and a recent-cases list. Server and paper text is escaped before display. A low-confidence flag on
  the prediction was first a raw 0.55 comparison, replaced on 2026-10-26 by the calibrated rule described below. The API now
  sends every time marked UTC (SQLite returns none, which made a browser show elapsed times 5.5 hours wrong).
  Two changes outside the page: the Phase 5 planner no longer says "consider clinical correlation or a follow-up
  study (for example CT)" for a low-confidence answer, because follow-up and clinical advice are out of scope and
  the page now shows that sentence; it says the system has no further input to request and the answer is uncertain.
  And `tests/conftest.py` points the default database and upload folder at a throwaway folder, so no test can touch
  `data/app.db` or `data/uploads` (an empty project database appeared once during this work. I first blamed a test
  run, but none could recreate it, and it was most likely made when the server was first started, because the app creates
  the database at startup. The guard is kept anyway).
  Tested in a real browser (`tests/test_analyze_page.py`, 14 tests, slow-marked, needs `pip install playwright`
  and `python -m playwright install chromium`, run with `-m slow`): real app and pipeline code, fake models. Running
  it for the first time found two real bugs, a button with the id "reset" shadowing the form's own reset method
  (so Clear did nothing) and a display rule overriding the hidden attribute (an empty X-ray placeholder), and a
  layout bug (the page was wider than a phone screen); each has a test. Deliberate breakages were caught: the
  Clear button, the hidden rule, unescaped sentences, unescaped paper text, and times without a zone. Then run
  once more through the page with the real models on the validation conflict case (true label Lung Cancer,
  predicted Pneumonia at 40 percent): no browser errors, all four images loaded, "the models disagree" shown,
  94 s of work on a loaded machine (2.5 GB RAM free, and the prediction appeared at 25 s because the models
  were still warming up when the case was submitted).
- Docker Desktop 4.49.0 was installed on 2026-10-25 with its data on D: (`D:\DockerData`), and
  checked: hello-world ran, a Postgres 16 container answered on host port 5433, and a container
  reached the host's Ollama through `host.docker.internal:11434` with no change to `OLLAMA_HOST`.
- Settings for compose (done 2026-10-25, `app/config.py`): `MEDREASON_OLLAMA_URL` and
  `MEDREASON_OLLAMA_MODEL` (the writer reads them at call time), `MEDREASON_QDRANT_URL` (unset means
  the embedded folder under `MEDREASON_QDRANT_PATH`, set means a Qdrant server, whose collections are
  named per embedding model). Checked for real: embedded retrieval still returns the same kind of
  passages, and a `qdrant/qdrant:v1.19.1` container was filled through `build_index` and read back
  through `search` and the `Retriever`. A server starts empty, so compose needs one
  `python -m app.rag.index --build` run against it.
- The storage layer was run against a real Postgres 16 container: all 10 storage tests passed, and
  the JSON columns are `jsonb`. To repeat it, start the container, set
  `MEDREASON_TEST_POSTGRES_URL=postgresql+psycopg://postgres:test@localhost:5433/postgres` and run
  `python -m pytest tests/test_database.py -m slow`. That run drops and recreates the three tables.
- `requirements.txt` now pins `torchxrayvision==1.5.4` and `psycopg[binary]==3.3.6`.
- Commands run from the tool here sometimes left a new container in the "Created" state instead of
  starting it (seen twice, not with the user's own terminal). `docker start <name>` fixed it.
- Step 5, observability (done 2026-10-26): `app/observability.py`, `app/health.py`, `app/rag/names.py`, and changes to
  `app/main.py`, `app/api/cases.py` and `app/pipelines/runner.py`. Design and reasons in `docs/PHASE7_DESIGN.md`.
  - Logs: the case id and a request id travel in context variables (`bind`), and a filter on the handler copies them onto
    every record, so helpers that know nothing about cases are tagged too. The worker thread and the request threads have
    separate contexts (tested). `MEDREASON_LOG_FORMAT` is `text` or `json`, `MEDREASON_LOG_LEVEL` filters. Only the
    `medreason` logger tree is configured; before this its INFO lines were invisible under uvicorn because no handler
    existed. The runner logs job and stage start and finish with seconds, a plain-ASGI middleware logs one line per request
    and returns `X-Request-ID` (the caller's value is kept only if it is 1 to 64 characters of letters, digits, dot,
    underscore, hyphen). A successful GET is DEBUG because the page polls; POST, DELETE and 4xx are INFO; 5xx is WARNING. A
    500 from an unhandled error does not carry the header, because the outer error middleware sends it. Logs never contain
    typed text or lab values (tested by searching the whole log). Start uvicorn with `--no-access-log` or its own request lines
    duplicate the app's. `create_app(setup_logging=False)` by default so tests do not change global logging; the module-level
    `app` turns it on.
  - `/api/health` is liveness only. `/api/ready` probes the database (a query), the worker thread, the evidence index and Ollama
    (reachable and the configured model pulled) at the same time, with a hard 5 s deadline. `down` (503) is database or worker
    missing, `degraded` (200) is Qdrant or Ollama missing, because the case then ends "partial" with the prediction and
    explanations intact. Details never contain a URL, connection string or path. Embedded Qdrant is checked by file presence only,
    since opening it would lock the retriever out, so an empty index would pass; a Qdrant server is asked for its point count.
    `/api/stats` gives cases by status and p50, p95 and worst seconds per stage from the events table (retention window only).
  - Tests: 37 new in `tests/test_observability.py`, against real local HTTP servers for Ollama and Qdrant rather than patched
    functions. 26 deliberate breakages were each caught by a test. The first round let one through (a file path in the embedded
    index's success message, where my test only looked at the failure message), and the test was strengthened. The 17 browser tests
    still pass with the middleware in front of the event stream.
  - Run for real (scratch database, real models, Ollama started and then stopped): with Ollama stopped, `/api/ready` took 4.0 s
    against a 2 s timeout, which led to the overall deadline. Chasing it found that on this laptop a connection to a closed
    loopback port takes 2.0 s to be refused, and Python tries `::1` before `127.0.0.1`, so every call to `localhost:11434` paid about
    2 s (2.05 s measured against 0.01 to 0.03 s). The default Ollama address is now `http://127.0.0.1:11434`. On the same
    validation case after warm-up, the mean time of a language-model call went from 2.97 s to 0.84 s (five calls each) and the writing
    stage from 15.1 s to 4.2 s; the total went from 94.8 s to 50.7 s, but part of that is the verifying stage (70.3 s then 23.2 s),
    which I have not explained and which is not network time. The retrieving stage is not comparable, because the lookup cache
    started empty after the restart. One case each, and the written text differs between runs, so read the call times as the
    measurement and the totals as indicative. With Ollama stopped, a case finished "partial" in 5 s: prediction and explanations kept, the
    writing failure logged as an ERROR with the case id and stage, and `/api/ready` reported `degraded` with HTTP 200.
  - A leftover to know about: `data/app.db` (last written 2026-10-25 by the user's own Analyze-page testing) holds those test
    cases. It is git-ignored and was not touched this session.
- Step 6, containers (done 2026-10-27): `Dockerfile` (stages base, test, runtime), `docker-compose.yml` (postgres 16, qdrant
  v1.19.1, api, and an opt-in `setup` service), `requirements-api.txt`, `.dockerignore`, `.env.example`, `.github/workflows/tests.yml`,
  `app/prefetch.py`, and the setting `MEDREASON_REBUILD_DASHBOARD`. Full notes, commands and measurements in `docs/DOCKER.md`.
  - The image holds code only. `experiments/` (trained models, result files) is mounted read-only, the pretrained downloads live in
    the `hf` and `xrv` volumes, the container runs with `HF_HUB_OFFLINE=1`, and the build context leaves out `data/` and every model
    or prediction file. All ports are published on 127.0.0.1 (no login). The database password comes from `.env` (git and docker
    ignore it) and has no default. The process runs as user `app`.
  - First-time setup is `docker compose up -d postgres qdrant`, then `docker compose run --rm -e HF_HUB_OFFLINE=0 setup python -m
    app.prefetch` (9.4 minutes when measured), then `docker compose run --rm setup python -m app.rag.index --build` (811 passages),
    then `docker compose up -d api`. A relative `-v ./data/corpus` on `docker compose run` silently mounted nothing, which is why
    the corpus mount lives in the `setup` service.
  - `docker build --target test .` runs the suite inside the image build (269 passed, 1 skipped, 13 deselected) against a context that
    excludes everything git ignores, so it doubles as a clean-checkout check and is the CI job. On GitHub the workflow has run
    on every push since (green, 412 passed in the log of the 2026-10-29 run). `tests/test_packaging.py` (21 tests) checks pin agreement between the two requirements files, that every
    import in `app/` and `explain/` is installed by the runtime file, loopback-only ports, no password default, read-only model
    mount, non-root user, and that `.dockerignore` covers what `.gitignore` does. 16 deliberate breakages were each caught.
  - Run for real: `/api/ready` was `ok` with the API reaching Ollama through `host.docker.internal` (13 ms) and Qdrant holding 811
    passages; a pneumonia validation case finished in 52.5 s (20 s of it the first language-model call while Ollama loaded its
    model) and a normal case in 6.5 s; the API used 1.77 GiB of 5 GiB; both cases and the uploaded image survived an API restart;
    stopping Ollama and Qdrant gave `degraded` (HTTP 200) in 3.8 s. Test cases were deleted through the API and the stack stopped
    with `docker compose down`, which keeps the volumes (models, index, empty uploads, Postgres).
  - Postgres `jsonb` does not keep key order, so `timings` comes back as writing, verifying, explaining, predicting, retrieving.
    The Analyze page looks stages up by name and is unaffected.
  - My own mistake worth keeping: I started a breakage-check script that edits source files while a docker build was running.
    BuildKit reads context files when each `COPY` runs, so that build could have baked in a broken file. I stopped the script before
    it mutated anything and rebuilt cleanly afterwards. Do not edit tracked files while a build is running.
- Step 7, end-to-end run and timing budget (done 2026-10-28): `scripts/run_e2e.py`, `tests/test_e2e_script.py` (12 tests, 10 deliberate
  breakages caught after one test was made asymmetric), results in `experiments/e2e/RESULTS.md` with aggregates in
  `experiments/e2e/{cold_start,warm}/summary.json` (the per-case `cases.json` holds study ids and is git-ignored).
  - Protocol: 16 validation studies chosen by rule and seed 42 (two agreeing pneumonia, two agreeing normal, two cancers fusion finds,
    two it misses, two wrong non-cancers, two with no labs on record, and one each with text only, image only, no X-ray, no
    indication), sent over HTTP to the compose stack, compared with the offline twin (stored per-modality probabilities pooled with the
    app's own rule for the inputs sent), then deleted from the server. Two passes: the first right after the containers and Ollama
    were started, the second straight after. The laptop was in ordinary use (3.6 GB free before Docker, 0.4 GB after the stack was up).
  - Result: 32 of 32 runs gave the same class as the offline twin (largest probability difference 0.0001), all explainers matched the
    stored prediction, all cases finished "done", and both passes gave identical probabilities, flags, sets and alerts. Prediction on
    screen at p50 0.53 s (p95 2.25 s). Whole case: Normal predictions need no evidence and take 1.5 to 2 s; with evidence the median
    was 19.2 s cold and 14.5 s warm, max 48.3 s and 26.9 s. Verifying is the largest stage, about 3.4 s per drafted sentence (p95
    about 7 s), which explains the earlier 70 s against 23 s. Language-model calls take about 0.9 s; only the first one after Ollama
    starts is slow (17.7 s). Retrieval is 1 to 8 s uncached and free when repeated.
  - The one thing that varies is the evidence note: same passages both times, but verified and dropped sentence counts changed in 2
    of 16 cases (25 verified and 7 dropped in the first pass, 23 and 9 in the second), the writer's non-reproducibility measured on a
    set. The "11 of 16 agree with the true label" figure is not an accuracy, because the set is stratified on purpose.
  - Limits: 16 studies, one laptop under load, percentiles on 16 values are rough, the warm pass is warm in three ways at once, and
    the NLI checker still runs on the CPU (a GPU build or a smaller model is the obvious speed-up and is untried).

### Phase 8, integration, testing, writeup (done 2026-10-29)

Public summary of everything below, written to be quoted from: `docs/RESULTS.md`. Each experiment has a `PLAN.md` written before it ran and a
`RESULTS.md`. Test count after this phase: 412 normal plus 43 slow-marked (375 and 38 before the out-of-scope work).

- The sealed test split was used ONCE, on 2026-10-29 (`scripts/eval_test_split.py`, `experiments/test_evaluation/`). Do not run `--score`
  again: it refuses while `results.json` exists, and a second look would no longer be a test. Do not retrain or retune the models, the
  fusion rule or the decision-rule levels and then quote a test number; any such change makes later test numbers optimistic and has to be
  reported as post-hoc. `scripts/submit_case.py` and the demo builder still refuse or never name the test split.
  - Protocol: plan and frozen-system hashes first; `--predict` (never loads the label column, resumable) then `--score` (labels read once). It
    was rehearsed on 120 validation studies in a scratch folder (a slow test) and every safeguard was broken on purpose and caught. The deployed
    system (pools only the inputs a case has) is the headline; the validation convention (all three always, blood fed all-missing) is reported
    beside it and reproduces the old 0.6133 on validation exactly.
  - Result (2,099 studies, 721 patients, 120 cancers): macro F1 **0.609** (0.559 to 0.654) against 0.613 on validation; F1 Normal 0.629 /
    Pneumonia 0.775 / Lung Cancer 0.421 (0.293 to 0.542); accuracy 0.704 (majority class 0.628); Lung Cancer ROC AUC 0.841 (0.767 to 0.904);
    calibration error 0.118 raw and 0.029 after temperature scaling; low-confidence rule answers 81.9 percent at 75.2 percent accuracy;
    prediction-set coverage 81.4 / 79.4 / 80.8 percent against a target of 80; cancer alert at the default setting flags 56.7 percent of
    cancers (44.0 to 69.0) at 11 alerts per 100 cases with 29 percent real; consistency verdicts agree / partial / conflict were right
    85.5 / 64.2 / 53.7 percent of the time (68 is the number the checker quotes for the middle group).
  - Post-hoc (not in the plan, labelled as such, `scripts/eval_test_posthoc.py`): paired gains in macro F1, validation then test. Fusion over
    text alone +0.104 then +0.049 (+0.014 to +0.081); text added to the other two +0.105 then +0.125; blood +0.051 then +0.029 (-0.001 to
    +0.057); image +0.026 then +0.005 (-0.013 to +0.021). So the validation estimates of the system's level were good, but the gain from
    combining inputs was about half as large on unseen patients and the image's contribution is indistinguishable from zero.
- Image backbone experiment (negative, `experiments/image_backbones/`): the NIH and PadChest weights (trained Nodule and Mass outputs, no MIMIC
  pretraining) and nih+mimic did not meet the pre-registered adoption rule; the current image model stays. Image Lung Cancer AUC: control 0.534,
  nih 0.515, pc 0.604 (paired +0.070, -0.003 to +0.144), nih+mimic 0.530. The leakage control was inconclusive (every non-MIMIC backbone scored
  lower on image Pneumonia F1, which inflation and domain shift both predict), so the caveat that the backbone may have seen our patients stays.
  Retraining the same recipe gives image macro F1 about 0.03 different (0.472 against 0.481 stored), so smaller image-alone differences mean
  nothing. The plan's Gate 0 compared with 0.50, the Phase 2 number; the stored out-of-fold predictions score 0.481, and the gate passed either way.
  Features are extracted locally on the CPU by `scripts/extract_image_backbone_features.py` (about an hour, resumable, git-ignored).
- Other-pathology measurement (`experiments/other_pathology/`, `scripts/eval_other_pathology.py`, and `scripts/build_final_dataset.py --excluded`
  which builds `data/processed/excluded.csv` from the 3,475 Excluded studies of 1,628 patients in neither training nor the sealed split): on 1,500
  of them the unchanged system answered 72.4 percent without a low-confidence flag, named Pneumonia 52.4 percent and Normal 38.1 percent, and
  confidence separated them from in-scope studies with AUROC 0.616 (0.589 to 0.640); the cancer alert fired on 18.2 percent against 10.0 percent
  of in-scope non-cancer studies. The system cannot say "none of these". Nothing was changed; the fix (an explicit other outcome or an
  out-of-scope detector trained on Excluded studies of training patients) would change the models and is listed as future work.
- Out-of-scope detector (negative, `experiments/out_of_scope/`, pre-registered): a gate meant to flag "possibly outside what this system can
  assess" from the three models' probabilities and the image backbone's 18 outputs, trained on in-scope training out-of-fold predictions against
  Excluded studies of training patients (`scripts/extract_scope_features.py`, `scripts/train_out_of_scope.py`; the Excluded tables come from
  `scripts/build_final_dataset.py --excluded --role train|val|none|unseen`, and test patients cannot be named). Cross-validated AUROC 0.738 (arm C),
  but on held-out patients (Excluded studies of patients in no split against in-scope validation) AUROC 0.573 (0.551 to 0.598), worse than the
  existing confidence signal (0.630), flagging 27.3 percent of ordinary studies against a target of 10 and 25.6 percent of true cancers. Not
  adopted, so no rules file or detector is written and the check is dormant. The post-hoc `clean_diagnostic` shows why: on rows with no seen-patient
  mismatch every arm is about 0.60 and the backbone outputs add nothing, so the 0.738 was an artifact (Excluded studies of training patients get
  predictions from models that saw those patients, the negatives are out-of-fold). Conclusion: the information separating other disease from the
  three classes is not in the system's existing outputs; closing the gap needs an explicit "other" outcome in the models. The plumbing stays:
  `app/agent/scope.py` (feature definition shared by training and serving, `available()`, `assess()`), `scope_of` in `app/pipelines/staged.py`, a
  private `_backbone_outputs` key from `explain.image.predict_image`, and the "Possibly outside what this system can assess" headline in
  `analyze.js`; `tests/test_out_of_scope.py` pins that nothing ships (`out_of_scope_rules.json` must not exist) and must be changed on purpose
  when a detector is ever adopted. Test count after this: see section 6.
- Non-X-ray guard (`app/xray_guard.py`, fitted by `scripts/build_xray_guard.py` on 3,000 training films, thresholds from 1,000 other training
  films, parameters in `app/data/xray_guard.json`): colour, flatness, structure beyond a tilted plane, saturated-pixel share, and a distance to a
  Gaussian model of 20 by 20 thumbnails. 0.35 percent of 2,315 validation films wrongly refused; resized, JPEG q30, RGB, rotated and blurred real
  films all pass; every required kind of non-film (noise, flat colours, gradients, checkerboards, text pages, shapes on white, smooth blobs, colour
  noise, colour gradients, colour photographs) is refused at least 95 percent. The first fitting failed the pre-registered criteria (gradients, 27
  percent of shapes, grayscale photographs), two cues were added from real films only, and the final numbers use a fresh seed for the negatives.
  Known limits: grayscale photographs are refused only half the time, a film upside down passes, it does not check chest or frontal view. The
  setting `MEDREASON_XRAY_GUARD` is true by default; `tests/conftest.py` turns it off for the suite.
- Demo (`scripts/build_demo_cases.py`, `app/api/examples.py`, the "Example case" menu in `app/static/analyze.js`): nine validation cases chosen
  by fixed rules (clear Normal, clear Pneumonia, a Lung Cancer found, the models disagreeing and right, low confidence, no labs, wrong but flagged,
  a cancer missed by the headline and caught by the alert, wrong and not flagged) frozen in `demo/cases/` (git-ignored, mounted read-only by
  compose, never in an image), plus two built-in non-X-ray pictures. The API never sends the study id or the true label. A slow test checks the
  live system still reproduces all nine and that the guard accepts all nine films. The old "Fill example values" button is gone: it put
  made-up values beside whatever X-ray the visitor chose.
- Mistakes worth remembering: a selection rule that uses each study once in profile order starves later profiles in tests unless the test
  records are isolated (hit three times); `frozen_system()` hashed files that a clean CI checkout lacks, caught only by thinking about the
  container build; and "uniform vector is neutral in the pool" is true for the winning class but not the probabilities (geometric mean over
  three models weights each by 1/3).

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

Phases 2, 3, 4, and 5 core are done (section 2). Fusion (logpool of the 3 modalities'
probabilities) reaches val macro F1 0.61. The three explainers and the agent core
(`app/agent/`) both work and are verified against real validation cases.

Next:

1. Phase 6 is done (the chain, its measurements and the dashboard page are in section 2).
   Keep the design rule: RAG explains, it never diagnoses, and it must not be able to change
   the predicted label. Open item carried to Phase 7: a faster checker (GPU or a smaller model).
2. Phase 7, backend and API is done (all seven steps), and so is Phase 8 (section 2). What is left is future work, in the order that
   matters: teach the models an "other" outcome (the largest honest gap; a gate on the existing outputs was tried and failed, see the
   out-of-scope detector bullet in Phase 8), replace or fine-tune the image
   model, test at a second site, and move the sentence checker to a GPU. All of it changes the models, so any new test number would be
   post-hoc now that the sealed split is spent. Original Phase 7 plan: wire `app/agent/orchestrator.run_case` into a FastAPI
   endpoint (upload an X-ray, type the indication, enter labs), add PostgreSQL for case
   storage, Docker compose for the full stack (API, Postgres, Qdrant, Ollama), and build
   the dashboard's placeholder "Analyze a case" page against that endpoint. The staged
   pipeline (step 2, done) already runs the prediction first, then the explanations, then the evidence.
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
- The test split was sealed until the one final evaluation, which is done (2026-10-29). It must not be scored again (see Phase 8).
- Tests: `python -m pytest` from the repo root (30 to 70 s depending on load, 412 tests, plus 43 slow-marked ones that are skipped by default: 26 browser tests
  of the Analyze page, 12 Postgres runs of the storage tests, the decision-rules rebuild check and three rehearsals against the real models, which need Playwright, a throwaway
  database URL or the training predictions). They use synthetic agent
  results and stub the retriever, Ollama and the NLI model, so they need no models or services
  and never read the git-ignored real cases. `tests/__init__.py` must stay, because a stray
  `tests` package in site-packages otherwise shadows the folder. Mark anything needing a model
  or service with `@pytest.mark.slow`. Add a test with every behaviour change, and break the
  code on purpose once to check the test fails.
- Any new labeling or dataset change means rerunning the pipeline (section 3), not
  hand-editing CSVs.
- Update [ROADMAP.md](ROADMAP.md) status markers and this file's section 2 when a phase
  moves.
- Git commits: plain message, no `Co-Authored-By` trailer and no AI attribution of any
  kind. The user's git identity is the sole author.
- Claude never runs `git push`; the user pushes. Do one improvement at a time and report after
  each. Ask before deleting or moving anything outside the repo.
- Prose in this file, ROADMAP.md, README files, `*.txt`, docstrings, and the eventual
  writeup: no em dashes, no en dashes as punctuation, and avoid the patterns commonly
  flagged as AI writing (puffery, forced rule of three, "it is worth noting", opening
  with "Moreover" or "Importantly", heavy bold and nested bullets). Plain sentences.

---

## 7. Persistent memory

Claude Code keeps cross-session memory at
`C:\Users\HP\.claude\projects\d--finalyearproject\memory\` (the project moved from C: to D: on
2026-10-25; the old `c--Users-HP-Desktop-finalyearproject` folder still holds the earlier
transcript). `MEMORY.md` is its index. Those notes and this file should agree. If they drift, the
code and `ROADMAP.md` win, then update both.

---

## 8. Machine and environment (read before touching Windows, WSL, Docker or the disks)

Full notes: [docs/ENVIRONMENT_AND_HANDOFF.md](docs/ENVIRONMENT_AND_HANDOFF.md). The short version:

- Windows 11 Pro 22H2 (build 22621), and the user will not upgrade it. Docker Desktop must be
  version 4.49.0, the last that supports this build. Newer versions refuse to install.
- Installing WSL once sent this laptop into a boot loop (hypervisor DMA remapping). It boots through
  a custom default entry with `hypervisoriommupolicy disable`. Make a restore point and export the
  boot entries before any virtualization, driver or boot change. Never touch hibernation or boot
  settings casually.
- C: is small (about 38 GB free on 2026-10-25), D: has about 280 GB, both on one NVMe drive. The
  project now lives at `D:\finalyearproject` (moved 2026-10-25). Docker Desktop 4.49.0 is
  installed and verified, with its data in `D:\DockerData` (`docker_data.vhdx` and `ext4.vhdx`). WSL
  is capped at 8 GB by `C:\Users\HP\.wslconfig`. The boot entries are backed up in `D:\bcd-backup`.
  Docker's automatic updates are turned off, so leave them off. Other projects live on D:, so never
  touch them.
- Native PostgreSQL 18 holds port 5432, so the container's Postgres uses 5433. Ollama runs on the
  Windows host, not in a container.
- A new path means new Claude memory. The `memory` folder was copied to
  `C:\Users\HP\.claude\projects\d--finalyearproject\` on 2026-10-25. Keep chat transcripts out of git.
- Prediction sets, a low-confidence headline and a cancer alert on the page (done 2026-10-26), answering the finding that
  averaging the three models loses cancers the text model finds. `app/agent/decision.py` applies rules from
  `app/data/decision_rules.json`, which `scripts/build_decision_rules.py` builds from the TRAIN out-of-fold predictions only
  (validation is used once, to report what each rule does, and the numbers are stored beside it; the test split is untouched).
  The rules are built separately for each of the seven combinations of inputs, because the pooled probabilities behave
  differently with two inputs than with three, so a case without labs uses different rules. Three outputs, none of which
  changes the predicted class (a test fuzzes this): (1) low confidence, when the calibrated top probability is below the level
  that keeps 80 percent of train cases (0.553 for all three inputs, the same abstain rule as `eval_uncertainty.py`; the earlier
  page flag compared the raw probability with 0.55, which was not what that analysis measured, and is replaced); (2) a
  prediction set from class-conditional conformal prediction (same functions as `eval_uncertainty.py`), 80 percent by default,
  90 percent by setting; (3) a cancer alert when the pooled Lung Cancer probability reaches a threshold chosen on train to
  catch 50, 60, 70 or 80 percent of true cancers, 60 percent by default. All-input validation results: confidence rule answers
  81 percent and is right on 76 percent of those; sets held the true class 79, 81 and 84 percent (Normal, Pneumonia, Lung
  Cancer) with 56 percent single-class; alert levels catch 56, 66, 76 and 84 percent of cancers at 8.9, 13.0, 21.0 and 33.0
  alerts per 100 cases, with 32, 26, 18 and 13 percent of alerts real (cancer is 5.1 percent of cases).
  An alert is offered only if its alerts were at least twice as often real as the base rate on validation (`MIN_ALERT_LIFT`):
  with the image alone they were 1.1 times (flags 42 of 100 cases, 6 percent real) and with blood alone 1.98 times, so
  neither shows one, and the page says so. Settings: `MEDREASON_PREDICTION_SET_COVERAGE` (0.8 or 0.9) and
  `MEDREASON_CANCER_ALERT_RECALL` (0.5, 0.6, 0.7 or 0.8). The page headlines "Uncertain" with the best guess below when
  confidence is low, lists what cannot be ruled out, and shows the alert as a banner (worded as the models' signal, not a
  diagnosis, with its measured recall and false-alarm cost) when it fires under a different headline. A class-conditional set
  can leave out the most probable class (near-even probabilities give only Lung Cancer, because the rare class is judged
  more leniently); the page says so. Honest result on the case that prompted this (study 50262066, true Lung Cancer, fused
  Pneumonia 40 percent): it is headlined Uncertain with all three classes in the set, but its fused cancer probability is
  0.22, so the default alert (0.30) does not fire; only the 80 percent level (0.19, 33 alerts per 100 cases) does. Tested with
  real models through the page, 8 deliberate breakages caught, 17 browser tests.
- Why three different X-rays all came out Pneumonia on the page (2026-10-25, checked with the real models): the example
  values are pneumonia-like text and labs (text 60 percent, blood 80 percent Pneumonia), and the image has the weakest
  vote in the fusion, so it is outvoted. The Normal X-ray alone gave Normal 69 percent but fused with the example values
  gave Pneumonia 49 percent ("two of three agree"). The image alone called the Lung Cancer film Pneumonia 39 percent
  (its Lung Cancer AUC is 0.53). With each case's own indication and labs the answers differ as they should: pneumonia
  92 percent, normal 92 percent (all agree), and the lung cancer case Pneumonia 40 percent flagged as a conflict (text
  Cancer 62 percent, blood Normal 93 percent). Use `scripts/submit_case.py --study-id ID` to send such a case to the
  running server. UI to do later: the single example button is misleading next to an uploaded X-ray (offer normal-like and
  cancer-like examples, or load a validation case), and the Phase 8b example-case loader covers it.
- A validation case with no labs on record (for example study 57386788) is sent without a blood input, so the blood
  model takes no part; the offline evaluations fed the blood model all-missing values in that situation (it said
  Normal 66 percent), which is a small difference between the evaluation setup and the app.
- Memory of the app, measured 2026-10-25 (`psutil`, one real validation case): Python with torch 0.5 GB, 1.2 GB after
  `warm_up()` loaded every model (17 s with the files cached), 2.3 GB after one full case (peak 2.3 GB). Ollama is a
  separate process: its llama3.2 runner took 1.2 GB of RAM and 2.4 GB of the 4 GB of GPU memory. So the app plus the
  language model need about 3.5 GB of RAM and 2.4 GB of GPU memory, and one worker means memory does not grow with
  the number of users (the queue grows instead). VS Code is not part of that: on this laptop it used 4.4 GB, of which
  the Pylance language server alone was 2.0 GB.
- Stopping Ollama: `Stop-Process -Force` on `ollama` alone leaves its model runner, `llama-server`, running as an
  orphan. Two of them sat for hours holding 2 GB of RAM and 3.9 of the 4 GB of GPU memory (which probably slowed
  later runs). Stop all three names: `Get-Process ollama, "ollama app", llama-server | Stop-Process`, then check
  `nvidia-smi` shows about 100 MiB used.
- Ollama has to be running on the host for the writer (`ollama serve` in its own PowerShell window;
  it listens on 127.0.0.1 only, and containers still reach it). The default address is `http://127.0.0.1:11434`, not
  `localhost`: on this laptop `localhost` costs 2 s per call (see Phase 7 step 5). Do not set `OLLAMA_HOST=0.0.0.0`,
  which would expose it to the network.
