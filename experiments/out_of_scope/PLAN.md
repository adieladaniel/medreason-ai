# Teaching the system to say "possibly outside what I can assess": plan, written before any detector was trained

Written 2026-10-29. The sealed test split is spent, so this work never touches it and none of its numbers can be called test numbers.

## Question

`experiments/other_pathology/RESULTS.md` found that the system cannot say "none of these": 72 percent of studies of other chest disease were
answered without a low-confidence flag, and the pooled confidence tells them from in-scope studies with an AUROC of only 0.62. The three class
probabilities carry little information about whether a film shows something outside the classes. Other signals the system already computes might:
the image backbone's 18 pathology outputs include Effusion, Edema, Pneumothorax, Cardiomegaly and Atelectasis, the very findings that make a study
"other". Can a small detector, trained on Excluded studies, flag them?

## What is built, and what is not

A detector that reads the system's own intermediate outputs and returns a score for "possibly outside the three classes". It is a gate beside the
existing safeguards. It never changes the predicted class, exactly like the low-confidence flag and the cancer alert. The classifier, the fusion and
the decision rules are unchanged, so the sealed-test results for the three-class system remain valid for the three-class system. What changes is
what the page can say about a case, and that is measured only on validation and on held-out Excluded studies.

## Data and the role of each part (all patient-level, nobody is in two roles)

| Studies | Patients | Role |
|---|---|---|
| in-scope training studies (their out-of-fold predictions) | 3,362 | negatives for training the detector |
| Excluded studies of training patients | 1,408 | positives for training the detector |
| in-scope validation studies | | negatives for the validation check and the final report |
| Excluded studies of validation patients | 336 | positives for one validation check |
| Excluded studies of patients in no split | 1,292 | positives for the final held-out evaluation, never used for any choice |
| Excluded studies of test patients | 310 | not touched |

Model and settings are chosen by patient-grouped 5-fold cross-validation on the training rows only. Validation is looked at once; the held-out
Excluded studies are used for the final report.

## Features (all available when a case is served, if there is an X-ray)

- The three models' class probabilities (9 numbers), whether labs and an indication were present, the pooled top probability and entropy.
- The image backbone's 18 pathology outputs (11 are informative for these weights).

Without an X-ray the detector does not run and the page says the scope check needs one.

## Arms (fixed now)

| Arm | Inputs | Model |
|---|---|---|
| A | class probabilities and pooled summaries only | logistic regression |
| B | A plus the 18 backbone outputs | logistic regression |
| C | A plus the 18 backbone outputs | gradient-boosted trees |

The arm with the best cross-validated AUROC on the training rows is taken; a tie within 0.005 goes to the simpler one. Not tried: a four-class
retrain of the three models, text features for the detector, other model families. Adding any after seeing results would be fishing.

## Operating point

The threshold flags 10 percent of the in-scope training studies (using their cross-validated scores). It is a setting, because how many false
flags are acceptable is a product decision, as it was for the cancer alert.

## Adoption rule (encoded in `scripts/train_out_of_scope.py`, function `adopt`)

Measured on the in-scope validation studies (negatives) and the Excluded studies of patients in no split (positives), with 95 percent
intervals that resample patients (2,000 draws). The gate is adopted only if all hold:

1. AUROC at least 0.75 and its lower bound at least 0.70 (the existing confidence signal scored 0.62).
2. At the operating point, at least 35 percent of the held-out Excluded studies are flagged, and no more than 13 percent of in-scope
   validation studies are (the target is 10).
3. Macro F1 on the in-scope validation studies that are not flagged is at least the macro F1 on all of them (0.613), and no more than 20
   percent of true Lung Cancer validation studies are flagged.

If it fails, the system is unchanged and this is reported as a negative result.

## Limits stated in advance

- The Excluded set is whatever was left over, not a designed set of other diseases, and its label does not say what the other condition was.
- The predictions for Excluded studies of training patients come from models that saw other studies of the same patients; the in-scope
  negatives are out-of-fold. The held-out Excluded studies are free of that mismatch, which is why the final report uses them.
- Same hospital, same labeller. An out-of-scope detector trained here says nothing about disease from another site.
- The in-scope test split is spent, so the cost of the gate on in-scope cases is measured on validation only.
