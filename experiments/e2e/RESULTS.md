# End-to-end run on the containerised service (Phase 7, step 7)

Run on 2026-10-28 with `scripts/run_e2e.py` against the docker compose stack (API, PostgreSQL 16, Qdrant server, Ollama on the
host with llama3.2). Aggregates are in `cold_start/summary.json` and `warm/summary.json`. The per-case records hold study
identifiers and are git-ignored.

## What was run

Sixteen validation studies, chosen by rule with a fixed seed (42), so the run can be repeated and the cases were not picked to
look good. Only the validation split was used; the test split is untouched.

| Category | Cases | How a study qualifies |
|---|---|---|
| agree, pneumonia | 2 | true Pneumonia, fusion says Pneumonia, labs on record |
| agree, normal | 2 | true Normal, fusion says Normal |
| cancer found by fusion | 2 | true Lung Cancer, fusion says Lung Cancer |
| cancer missed by fusion | 2 | true Lung Cancer, fusion says something else |
| wrong, not cancer | 2 | true Normal or Pneumonia, fusion wrong |
| no labs on record | 2 | all 17 lab values missing, so no blood input is sent |
| text only, image only, no X-ray, no indication | 1 each | a correct pneumonia study with inputs taken away |

Each case was sent over HTTP exactly as the Analyze page sends it, read back, compared with its offline twin, and deleted from the
server. The offline twin is the stored per-modality probabilities (`experiments/*_baseline/val_predictions.csv`) pooled with the
app's own rule for the inputs that were actually sent. Two passes on the same server: the first right after the containers and
Ollama were started (cold), the second straight after (warm: the language model loaded, the retrieval cache filled).

Conditions: the laptop was in ordinary use. Free host memory was 3.6 GB before Docker started and 0.4 GB after the stack was up
(2.0 GB at the end), with Chrome, VS Code and other programs open. These are loaded-laptop numbers, not best-case ones.

## Does the deployed pipeline reproduce the evaluated models?

Yes, on every case.

- Same predicted class as the offline twin: 32 of 32 runs. Largest probability difference from the offline numbers: 0.0001.
- The three explainers matched the stored prediction in 32 of 32 runs.
- The two passes gave identical probabilities, identical low-confidence flags, prediction sets and cancer alerts on all 16 cases.
  The prediction side of the system is deterministic.
- All 32 cases finished with status "done"; none failed or ended "partial".

## Timing budget

Seconds. Cold is the first pass, warm the second. "All cases" includes the cases where nothing needs explaining.

| Stage | Cold p50 | Cold p95 | Cold max | Warm p50 | Warm p95 | Warm max |
|---|---|---|---|---|---|---|
| predicting | 0.10 | 0.73 | 1.74 | 0.09 | 0.13 | 0.14 |
| explaining | 1.17 | 6.63 | 7.21 | 1.06 | 3.16 | 6.40 |
| retrieving | 1.15 | 7.45 | 7.75 | 0.00 | 0.00 | 0.00 |
| writing | 3.49 | 9.63 | 21.01 | 3.20 | 4.80 | 5.04 |
| verifying | 7.04 | 18.28 | 20.59 | 6.21 | 18.01 | 18.54 |
| whole case | 13.9 | 36.3 | 48.3 | 9.7 | 25.4 | 26.9 |
| prediction on screen | 0.53 | 1.85 | 2.25 | 0.52 | 0.62 | 0.63 |

By kind of case (whole case, seconds):

| | Cold | Warm |
|---|---|---|
| prediction is Normal, so no evidence is requested (4 cases) | median 1.8, max 2.0 | median 1.6, max 2.0 |
| evidence note needed (12 cases) | median 19.2, max 48.3 | median 14.5, max 26.9 |

What this says in plain terms: the prediction is on screen in about half a second, and a full note with verified citations takes
about 15 to 20 seconds, up to about 50 seconds for the first case after a start.

Where the time goes:
- Verifying is the largest stage. It costs about 3.4 seconds per drafted sentence (median 3.6 cold and 3.35 warm, 95th percentile
  7.0 and 6.2), so a case with five drafted sentences spends about 17 seconds there. This explains the earlier puzzle of 70 s
  against 23 s for similar cases: it is the number of sentences and the load, not a fault.
- Language-model calls are fast now: median 0.91 s cold and 0.83 s warm. The one slow call was the first (17.7 s) while Ollama
  loaded the model into the GPU. That is the whole cold-start cost of the writing stage (max 21.0 s against 5.0 s warm).
- Retrieval costs 1 to 8 seconds for a lookup the first time and nothing when repeated, because lookups are cached in the process.
  The warm pass saw only repeats, so its 0.00 is the cache, not the retriever.

## Findings about the system, not just its speed

- The evidence note is the only part that varies between runs. The same 16 cases retrieved the same passages both times, but the
  count of verified and dropped sentences changed in 2 of 16 cases (4 verified, 0 dropped became 3 and 1; 2 and 1 became 1 and 2).
  Totals were 25 verified and 7 dropped in the cold pass, 23 and 9 in the warm pass. This is the writer not being reproducible at
  temperature 0 (noted in step 2), now measured on a set of cases: the shown note can differ between runs of the same case, and
  the checker decides what reaches the screen.
- 11 of 16 cases ended with at least one verified background sentence; 5 had none (four predicted Normal, which asks for nothing,
  and the text-only case, whose single passage produced no verified sentence).
- The decision layer's output on the five true cancers in the set (the two found by fusion, the two it missed, and one study with
  no labs on record that fusion found): the alert fired on four, including one of the two that fusion missed, which also showed
  Lung Cancer in its prediction set. It stayed quiet on the other missed cancer, which was headlined Pneumonia with a single-class
  set. Five cases cannot say how often that happens; the validation-wide numbers are in experiments/test_evaluation/RESULTS.md.
- The 11 of 16 cases whose prediction agrees with the true label is not an accuracy. The set was stratified on purpose (ten were
  chosen because fusion gets them right, four because it gets them wrong, and so on).

## Limits

- Sixteen studies, two passes, one laptop under ordinary load. The percentiles on 16 values are rough, and the 95th percentile is
  close to the maximum.
- The warm pass is warm in three ways at once (model loaded, cache filled, files cached), so it shows the steady state, not a
  second independent cold measurement. A start after a reboot will be slower than the cold pass here (the OS file cache was warm).
- The language model and the NLI checker run on the CPU in the container; Ollama uses the GPU on the host. Moving the checker to
  a GPU build or a smaller model is the obvious speed-up for the verifying stage and has not been tried.
- Nothing here measures how good the written notes are. The checker's error rates are in experiments/rag_retrieval/checker_eval.json and docs/RESULTS.md.
- The offline twins use the stored probabilities, so a match shows the deployed models and pooling reproduce the evaluation,
  not that the evaluation itself is right.
