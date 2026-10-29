# Out-of-scope detector: results

Run on 2026-10-29 with `scripts/extract_scope_features.py` and `scripts/train_out_of_scope.py`, following `PLAN.md` (written before any
detector was trained). The sealed test split was not opened. Numbers are in `results.json`.

## Outcome

**Not adopted.** The detector failed four of the five conditions of the adoption rule fixed in advance, so nothing was written for the app to read
and the system is unchanged. The scope-check plumbing that was built (`app/agent/scope.py`, a hook in the prediction stage, a headline on the page)
stays in the code, tested and dormant: with no fitted detector a case pays nothing for it.

## What happened

Training rows: 11,039 in-scope training studies (out-of-fold predictions) against 5,369 Excluded studies of training patients, 3,362 patients.
Patient-grouped 5-fold cross-validation on those rows:

| Arm | Inputs | Cross-validated AUROC |
|---|---|---|
| A | class probabilities and pooled summaries | 0.610 |
| B | A plus the image backbone's 18 outputs, logistic | 0.620 |
| C | A plus the 18 outputs, gradient-boosted trees | **0.738** |

Arm C was chosen and judged once on patients it had never seen: the 2,241 Excluded studies of patients in no split at all against the 2,315
in-scope validation studies.

| | Required | Measured (95 percent interval) |
|---|---|---|
| AUROC | at least 0.75, lower bound at least 0.70 | **0.573** (0.551 to 0.598) |
| Excluded studies flagged | at least 35 percent | 34.5 percent (32.3 to 36.8) |
| in-scope validation studies flagged | at most 13 percent (the target was 10) | **27.3 percent** (24.6 to 30.0) |
| macro F1 of unflagged against all in-scope validation | unflagged at least as high | 0.630 against 0.613, met |
| true Lung Cancer validation studies flagged | at most 20 percent | **25.6 percent** (17.6 to 35.6) |

For context, the system's existing confidence signal scored AUROC 0.630 (0.606 to 0.653) on the same data, so the detector was worse than
what was already there. The Excluded studies of validation patients, checked first, gave AUROC 0.604 and a catch rate of 38.1 percent.

## Why: the cross-validation was fooled by how the training rows were built

The cross-validated 0.738 against the held-out 0.573 is a large gap on patients of the same kind. The plan had named the cause as a risk: the
in-scope negatives are out-of-fold, but the Excluded studies of training patients get predictions from models that had already seen those
patients' other studies. A flexible detector can learn "was this patient seen by the models" instead of "is this another disease".

The post-hoc check (labelled as such, `clean_diagnostic` in the script) repeats the comparison on rows with no such mismatch: in-scope
validation studies against Excluded studies of validation patients, every prediction made on patients the models had never seen, cross-validated
by patient (2,315 and 1,234 studies, 720 patients).

| Arm | Cross-validated AUROC on mismatch-free rows | In-scope against the no-split Excluded studies |
|---|---|---|
| A | 0.600 | 0.543 |
| B | 0.602 | 0.564 |
| C | 0.604 | 0.594 |

On clean rows every arm sits near 0.60 and the backbone's findings add nothing (0.600 to 0.604). So the 0.738 was an artifact, and the real
conclusion is stronger than "this detector failed": **the information that separates other chest disease from the three classes is not in what
the system already computes.** Likely reasons, which this experiment did not test: the backbone's eleven outputs (effusion, edema, atelectasis
and the rest) also occur in pneumonia, and many Excluded studies may have findings (devices, lines, post-operative changes) that none of the
eleven covers. Whatever the reason, a detector built on these outputs does not supply the missing "other" outcome.

## What it means for the project

- The gap measured in `experiments/other_pathology/RESULTS.md` stays open and is now better understood. It cannot be closed by a gate on the
  existing outputs, and the documentation says so rather than shipping a gate that flags a quarter of ordinary studies.
- A real fix needs new information: a model trained with an explicit "other" outcome on the three inputs (retraining the image head, blood and text
  models and the fusion with a fourth class, using properly out-of-fold features for all four classes), or a separate model of non-target findings
  trained from the report text. Both change the models and have no unspent test data to be judged on beyond what this experiment already used.
- A methodological lesson worth keeping: when positives come from a different population than the negatives in a way the model can see (here,
  seen against unseen patients), cross-validation will look good and the held-out set will say otherwise. The held-out set was reserved for
  exactly this reason.

## Limits

- The Excluded set is whatever was left over, not a designed set of other diseases, and its label does not say what the condition was.
- One detector family on one feature set; the plan fixed three arms in advance and no others were tried.
- The clean diagnostic is post-hoc. It was run because the held-out result disagreed with cross-validation, and it uses data (Excluded studies of
  validation patients) that was a first check in the plan, not training data.
- 11,039 in-scope and 5,369 Excluded training studies come from one hospital system.
