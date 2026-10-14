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
| 7. Backend and API | not started | |
| 8. Integration, testing, writeup | not started | |

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
- Latency: the cross-encoder is essentially all of it. Reranking 5 passages takes about 1 to
  1.5 s per query now; earlier measurements on the same machine were 3 to 6 times slower, so
  expect 1 to 8 s. Depth 5, 10, and 20 gave the same quality. Fewer passages clear the
  threshold at depth 5 (often 1 to 3), which is the trade-off.
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
  words and take 3 to 4 s per item. llama3.2 stays. Limits: 41 items, so the gap is about four
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
  These checks should be repeated once, unchanged, on the test split at the end.

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

Phases 2, 3, 4, and 5 core are done (section 2). Fusion (logpool of the 3 modalities'
probabilities) reaches val macro F1 0.61. The three explainers and the agent core
(`app/agent/`) both work and are verified against real validation cases.

Next:

1. Phase 6 is done (the chain, its measurements and the dashboard page are in section 2).
   Keep the design rule: RAG explains, it never diagnoses, and it must not be able to change
   the predicted label. Open item carried to Phase 7: a faster checker (GPU or a smaller model).
2. Phase 7, backend and API. Wire `app/agent/orchestrator.run_case` into a FastAPI
   endpoint (upload an X-ray, type the indication, enter labs), add PostgreSQL for case
   storage, Docker compose for the full stack (API, Postgres, Qdrant, Ollama), and build
   the dashboard's placeholder "Analyze a case" page against that endpoint. Watch out for
   the text explainer's 60 to 90 second SHAP run; either move it off the request path or
   show the prediction and image and blood explanations first, then stream text after.
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
- Tests: `python -m pytest` from the repo root (about 15 s, 64 tests). They use synthetic agent
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
