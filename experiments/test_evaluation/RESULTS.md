# The sealed test evaluation: results

Run on 2026-10-29 with `scripts/eval_test_split.py`, following `PLAN.md` (written before the split was opened). 2,099 studies from 721
patients, none of whom appear in training or validation; 120 Lung Cancer studies. The system was frozen first (`frozen_system.json`
records the hashes of the models, the backbone weights, the decision rules and the prediction path, and the scoring step checked they had
not changed). The test split was scored once; the script now refuses to run again. Numbers are in `results.json`; the only addition after
scoring is the post-hoc table below, labelled as such.

Intervals are 95 percent, resampling patients (2,000 draws).

## Headline

The system as deployed (it pools only the inputs a case has, as the app does) scores **macro F1 0.609 (0.559 to 0.654)** on the sealed
test split, against 0.613 on validation. The validation figure was a little optimistic by construction (the fusion rule was chosen
among five on validation), and the difference is far smaller than the interval.

| | Validation | Test (sealed) |
|---|---|---|
| macro F1 | 0.613 | **0.609** (0.559 to 0.654) |
| F1 Normal | 0.615 | 0.629 (0.590 to 0.668) |
| F1 Pneumonia | 0.797 | 0.775 (0.745 to 0.799) |
| F1 Lung Cancer | 0.426 | 0.421 (0.293 to 0.542) |
| accuracy | 0.719 | 0.704 (0.676 to 0.729) |
| accuracy of always saying the majority class | 0.663 | 0.628 |
| Lung Cancer ROC AUC | 0.853 | 0.841 (0.767 to 0.904) |

Accuracy is only 7.6 points above the majority-class rate (always answering Pneumonia); macro F1, which gives the rare class the same
weight as the common ones, is the figure to quote. Confusion matrix on test (rows are the truth, columns the answer: Normal,
Pneumonia, Lung Cancer): Normal 446, 188, 26; Pneumonia 282, 977, 60; Lung Cancer 29, 36, 55.

The all-inputs "validation convention" (every model pooled for every case, as the 0.613 was computed) gives 0.610 (0.561 to 0.655) on
test. Both ways of forming the answer are in `results.json` and agree to within 0.003.

## Each input alone and in pairs (macro F1)

| | Validation | Test |
|---|---|---|
| image alone | 0.481 | 0.460 (0.432 to 0.488) |
| blood alone | 0.461 | 0.446 (0.409 to 0.485) |
| text alone | 0.509 | 0.561 (0.519 to 0.602) |
| image + blood | 0.509 | 0.484 |
| image + text | 0.560 | 0.581 |
| blood + text | 0.586 | 0.605 |
| all three | 0.613 | 0.609 |
| Lung Cancer ROC AUC: image / blood / text | 0.527 / 0.718 / 0.842 | 0.597 (0.514 to 0.675) / 0.696 / 0.832 |

The ordering from validation holds: all three are at least as good as any pair, text is the strongest single input, and the image adds
least. The text model scored higher on test than on validation (0.561 against 0.509), which is within what sampling noise can do with this
many studies but is not explained.

## Post-hoc: what each input adds when the other two are present

Not in the plan. Added because the table above made the gain from fusion look smaller on test, and the unpaired intervals overlap.
Differences in macro F1, paired on the same resampled patients (`scripts/eval_test_posthoc.py`, `posthoc.json`). Computed from
the already-scored predictions; nothing in the system depends on it.

| Adding this input | Validation | Test |
|---|---|---|
| fusion over text alone | +0.104 (+0.074 to +0.133) | +0.049 (+0.014 to +0.081) |
| text (to image and blood) | +0.105 (+0.063 to +0.143) | +0.125 (+0.083 to +0.163) |
| blood (to image and text) | +0.051 (+0.023 to +0.078) | +0.029 (-0.001 to +0.057) |
| image (to blood and text) | +0.026 (+0.011 to +0.042) | +0.005 (-0.013 to +0.021) |

This is the most important qualification of the project's claims. Fusion still beats the best single input on unseen patients, but by
about half what validation suggested. The text input carries the result. Blood helps, borderline on test. **The image's contribution cannot
be distinguished from zero on test**, where on validation it was small but clearly above zero. The validation figures for these four
quantities are reproduced exactly by this code (they match the earlier analysis in `scripts/eval_dominance.py`).

## Calibration and the safeguards on unseen patients

| | Validation | Test |
|---|---|---|
| expected calibration error, raw pooled probabilities | 0.142 | 0.118 (0.094 to 0.144) |
| expected calibration error after the stored temperature scaling | 0.026 | **0.029** (0.019 to 0.057) |
| Brier score, raw / calibrated | 0.430 / 0.397 | 0.430 / 0.404 |
| answered without the low-confidence flag | 81.5 percent | 81.9 percent (79.8 to 83.7) |
| accuracy on the cases it answers | 76.5 percent | 75.2 percent (72.2 to 77.8) |
| macro F1 on the cases it answers | 0.661 | 0.669 (0.609 to 0.719) |
| true cancers it declines to answer | 31.6 percent | 30.8 percent (21.7 to 41.7) |

The calibration learned from training predictions transferred: the pooled probabilities are under-confident, the temperature scaling fixes
it (error 0.118 to 0.029), and declining the least confident fifth of cases raises accuracy on the rest, on patients the rules never saw.

Prediction sets at the default 80 percent level: the true class was in the set for **81.4 percent of Normal, 79.4 percent of Pneumonia and
80.8 percent (69.6 to 89.6) of Lung Cancer studies**, against a target of 80 for each. The coverage guarantee held on unseen patients. 58.6
percent of sets name a single class and the mean size is 1.43.

## The cancer alert

Rows are the four settings of `MEDREASON_CANCER_ALERT_RECALL` (the share of true cancers the alert was set to catch on training data).

| Setting | Share of cancers flagged (test) | Alerts per 100 cases (test) | Share of alerts that are real (test) | Validation: flagged / per 100 / real |
|---|---|---|---|---|
| 0.5 | 51.7 percent (38.6 to 65.0) | 8.2 | 35.8 percent | 54.7 / 8.8 / 31.5 |
| 0.6 (default) | 56.7 percent (44.0 to 69.0) | 11.1 | 29.3 percent | 65.0 / 12.4 / 26.4 |
| 0.7 | 70.8 percent (59.6 to 81.3) | 18.0 | 22.5 percent | 73.5 / 19.5 / 19.0 |
| 0.8 | 76.7 percent (65.6 to 85.9) | 29.1 | 15.1 percent | 80.3 / 30.6 / 13.3 |

The false-alarm cost behaved as measured on validation. The catch rate was a little lower than targeted at the default (56.7 against 60
percent; the interval includes 60), as expected for rules fitted on one set of patients and applied to another. Cancer is 5.7 percent of the
test studies.

## The consistency check

Accuracy of the pooled answer, by what the consistency check said (deployed system):

| Verdict | Cases on test | Correct on test | Correct on validation |
|---|---|---|---|
| all three agree | 740 | 85.5 percent | 86.1 percent |
| two agree | 1,102 | 64.2 percent | 67.1 percent |
| all three differ | 257 | 53.7 percent | 56.0 percent |

The sentences the system says to users ("cases like this are correct about 86, 68 and 53 percent of the time") are supported on unseen
patients to within about four points (the largest gap is the middle group: 68 quoted, 64.2 measured).

## How to read it

- The validation numbers were good estimates. The drop is 0.004 in macro F1, and the safeguards (calibration, prediction sets, the
  answered-only accuracy) behaved on unseen patients as they were measured to behave.
- What did not carry over is the size of the gain from combining inputs, in particular the image's.
- Every Lung Cancer number is wide (120 cases). F1 0.42 means the system finds cancer imperfectly and raises many false alarms; the
  separate alert trades that off explicitly.

## Limits

- Same hospital system, same linking rule and labeller as training. This is not external validation, and nothing here says how the system
  would behave at another site or with another X-ray scanner.
- Labels come from admission-level codes and report mentions, so they are noisy; scores measure agreement with those labels.
- The image backbone's pretraining may include some of these patients (an open caveat the image-backbone experiment could not clear).
- The decision-rule levels and the fusion rule were chosen on validation; test is the unbiased check, and it is now used. Changing the system
  because of it would make the next number optimistic, so the system was not changed.
- The other-pathology measurement (`experiments/other_pathology/RESULTS.md`) is separate: this evaluation, like training, contains only
  the three classes.
