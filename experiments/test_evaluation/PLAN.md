# Sealed test evaluation: plan, written before the test split was opened

Written 2026-10-29. This is the one run of the whole system on `data/processed/test.csv` (2,099 studies, 15 percent of the data,
patients who appear in neither training nor validation). Nothing about the system may change because of what it shows.

## What is frozen

The system as it stands after the image-backbone experiment (negative: nothing adopted) and the other-pathology measurement
(measurement only: nothing changed). `frozen_system.json` records, before any prediction is made, the SHA-256 of the three trained
models, the backbone weights, the decision rules and the source files of the prediction path, plus the git commit. The scoring step
checks they have not changed since the predictions were made.

## How it is run

Two commands, in this order.

1. `python scripts/eval_test_split.py --predict`. For each test study, builds the inputs the app would build (the X-ray, the labs on
   record, the INDICATION section of the report, never the full report) and runs the three trained models. It never loads the label
   column. It saves each model's probabilities and which inputs were present, and it can be stopped and resumed, because the
   predictions do not depend on the labels.
2. `python scripts/eval_test_split.py --score`. Reads the labels, computes everything below, writes `results.json`, and refuses to
   run again once that file exists. A second look at the test split would no longer be a test.

## What is reported, and how the headline is chosen

Two ways of forming the fused answer are reported side by side, and neither is dropped:

- **Deployed system (the headline).** Only the inputs a case actually has are pooled, as the app does: no labs on record means no
  blood model, an empty indication means no text model. The low-confidence rule, prediction set and cancer alert use the rules built
  for that combination of inputs.
- **Validation convention.** All three models are always pooled, with the blood model fed all-missing values when no labs exist.
  This is how the validation numbers in the notes (macro F1 0.6133) were computed, so it is the like-for-like comparison.

The same code scores the validation split from its stored predictions, so the table of validation against test compares like with like.

Reported for both, with 95 percent cluster bootstrap intervals over patients (2,000 draws, seed 42):

1. Fused macro F1, per-class F1, accuracy, confusion matrix, Lung Cancer ROC AUC. The majority-class accuracy for reference.
2. Each model alone (image, blood, text) and the pairs, to check the ordering found on validation (text most, image least).
3. Calibration: expected calibration error and Brier score, raw and after the temperature scaling in the rules file.
4. The low-confidence rule: share of cases answered, accuracy and macro F1 on those, share of true cancers declined.
5. Prediction sets at the default 80 percent level: coverage for each true class (target 80), share with a single class, mean size.
6. The cancer alert at all four levels: share of true cancers flagged, alerts per 100 cases, share of alerts that are real.
7. The consistency checker: how often it says agree, partial or conflict, and accuracy within each, against the accuracy by status that
   it quotes to users (from validation).

## How the result will be read

- There is no pass mark. The numbers are reported as they come, next to validation, with their intervals.
- The validation figures were used to choose the fusion rule among five and to set decision-rule levels, so they are slightly
  optimistic. A modest drop on test is expected and is not an alarm. A drop larger than the intervals would be reported as such and
  investigated for a cause (for example a difference between the splits), but the models would not be retrained to recover it.
- Nothing is tuned afterwards. If a flaw in the evaluation code (not in the models) is found after scoring, it is fixed, the fix is
  stated, and the result is shown both ways. The score is not to be re-run to get a better number.

## Limits stated in advance

- 117 validation cancers became about as many in test; every Lung Cancer number is wide.
- Same hospital system, same linking rule, same labeller as training: this is not external validation.
- The image backbone's pretraining may include some of these patients (the open caveat); the test split cannot clear it.
