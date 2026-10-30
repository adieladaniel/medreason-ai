# Version 2: teaching the models an "other" outcome. Results

Run on 2026-10-31 following `PLAN.md`, which was written before any model was trained and amended once, after step 0, with facts it had missed. The
sealed test split was not opened. Numbers are in `results.json`, `arm_choice.json`, `oof_summary.json` and `step0.json`.

## Outcome

**Not adopted.** The chosen arm failed two of the five conditions of the rule fixed in advance, so version 1 stays as it is, nothing in the app
changes, and the system still cannot reliably say that a study is outside Normal, Pneumonia and Lung Cancer. Training the models with an explicit
fourth class made the flag a little better than the confidence signal the system already has, but not distinguishably so, and far short of the bar.

## What was done

- Step 0 checked the inputs. The cached image outputs were aligned with their tables, the Excluded class is 99.9 percent reports that state some
  finding other than pneumonia or cancer, and the held-out patients are by construction patients with no in-scope study.
- Step 1 built the four-class tables: 16,408 training studies (3,362 patients) with a patient-grouped fold, 3,549 validation studies (720 patients, 1,234 of them
  Excluded) and 2,241 held-out Excluded studies of 1,292 patients.
- Step 2 produced out-of-fold predictions for both arms and chose between them on training data only. The saved version 1 models reproduce the stored
  version 1 validation predictions (largest difference 2e-7), so the inputs match the audited system.
- Step 3 fused them, fixed the operating point and wrote the adoption function, rehearsed on one training fold.
- Step 4 froze the choice and its 9 prediction files by hash, and scored the validation and held-out rows once.

## The arm

| | Fused "other" score, out-of-fold AUROC on 16,408 training studies |
|---|---|
| J: each model retrained on four classes | 0.650 (blood 0.541, text 0.651, image 0.567) |
| T: version 1 models unchanged plus a binary model per input | **0.657** (blood 0.547, text 0.654, image 0.570) |

T was chosen. It is also the cautious design: its three-class answers are the version 1 answers, so the per-class validation scores are identical to
version 1 (Normal F1 0.618, Pneumonia 0.797, Lung Cancer 0.425, macro F1 0.613). The arm J was never scored on validation or held-out data, because the
protocol allows one arm to be scored; its training-time score gave no sign that it would have passed. The image head's three seeds agreed closely on
this task (AUROC of the other score 0.566 to 0.568, in-scope macro F1 0.482 to 0.489).

## The one scoring (validation in-scope studies as negatives, held-out Excluded studies as positives, intervals resample patients)

The operating point flags 10 percent of in-scope training studies by their out-of-fold scores (threshold 0.576, temperature 0.55).

| Condition | Required | Measured (95 percent interval) | Met |
|---|---|---|---|
| ROC AUC of the other score | at least 0.75, lower bound at least 0.70 | **0.653** (0.627 to 0.680) | no |
| Other studies flagged / in-scope studies flagged | at least 35 percent / at most 13 percent | **20.8** percent (18.4 to 23.2) / 6.9 percent (5.5 to 8.4) | no |
| Three-class macro F1, in-scope validation | at least 0.583 | 0.613 (0.568 to 0.650) | yes |
| True Lung Cancer studies flagged | at most 20 percent | 3.4 percent (0.7 to 7.2) | yes |
| Calibration error after temperature scaling | at most 0.05 | 0.036 (0.025 to 0.057) | yes |

The three conditions that passed are the ones a design that leaves version 1 intact passes almost by construction. The two that measure whether the
flag works did not pass.

## Supporting numbers (not part of the rule)

- The existing confidence signal (one minus the top probability) scored 0.632 on the same patients. This arm minus the signal is +0.021 (-0.008 to
  +0.052), so the two cannot be told apart.
- Using the Excluded studies of validation patients as positives instead gave 0.678; the held-out patients gave 0.653. Step 0 had warned that these two
  groups need not look alike. They differ by 0.025, which is small, though a small difference does not show that none exists.
- Other settings of the flag. At 5 percent of in-scope training studies: 11.1 percent of held-out studies flagged, 3.2 percent of in-scope validation
  studies. At 20 percent: 34.2 percent flagged, 14.7 percent of in-scope. Neither setting meets both limits. The curve is simply shallow.
- By inputs present. With all three inputs (2,189 in-scope validation, 2,030 held-out) 7.0 and 22.2 percent were flagged. With no labs on record, none
  of 57 in-scope and none of 126 held-out studies were flagged. With no indication, 9.7 percent of 72 and 18.4 percent of 87. So the flag never
  fired when labs were missing, which fits the idea that much of its signal comes from the labs, but the cause was not isolated and the groups are small.

## What this does and does not show

It shows that three frozen feature sets (the image backbone's 18 outputs, a frozen clinical-text embedding of the reason for the exam, and 19 lab
features), each given the Excluded studies as a class and fused by geometric mean, carry little information about which studies are "other". Per
input the out-of-fold AUROC was between 0.54 and 0.65.

It does not show that no model could do it. Reasons it may be hard here, none of them tested: the Excluded class is mixed, and the findings in it
(effusion, edema, atelectasis, cardiomegaly) often occur alongside pneumonia in the same patients; the training positives are other studies of patients
who also have in-scope studies, so patient-level signal in the labs does not separate them; the backbone outputs are frozen and 7 of its 18 are untrained
constants; and the indication states a reason for the exam, not a finding. Learning features for this task, for example by fine-tuning the image
model on report-derived finding labels, is the untried direction, and it would change the models.

## Differences from the plan, and limits

- The plan said the paired comparison would be with the failed gate. The gate's scores were not saved, so the comparison is with the existing
  confidence signal on the same patients instead. The gate's AUROC (0.573) is quoted from its own results.
- The planned "share of No Finding Excluded studies flagged" was dropped, because step 0 found those are 0.1 percent of the class.
- Step 0 expected the embedding to take about 36 minutes. With texts sorted by length it took 575 seconds; the sorted batches reproduce the cached
  version 1 embeddings to 1.9e-6.
- Recipes were copied from version 1 and not tuned. A tuned model might do better; that would be a new experiment with its own plan.
- The in-scope validation studies were used in version 1 choices. Here they were looked at once, which is the most that can be done without new data.
- One hospital system, and the Excluded class is whatever the labelling rules left over.
- The test split stays spent. Nothing here is a test number.

## Decision

Version 1 is unchanged and remains the system. The scope-check plumbing stays dormant, as after the first attempt, and the page does not claim that it
can detect studies outside its three classes. The dashboard's Reliability page now shows both attempts next to the confidence signal. Step 5 of the plan
(integration) does not apply, since it was only for an adopted detector.
