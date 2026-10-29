# What the system does with disease it has no class for: results

Run on 2026-10-29 with `scripts/eval_other_pathology.py`, following `PLAN.md` (written before the data was built). The unchanged
system, through the app's own prediction path, on 1,500 "Excluded" studies (952 patients) that belong to patients who are in neither
the training split nor the sealed test split. 1,386 had all three inputs, 58 had no indication, 56 had no labs. The comparison group
is the 2,315 in-scope validation studies, scored by the same pooling and decision code. Intervals are 95 percent, resampling
patients. Aggregates are in `results.json`; the per-study file holds patient identifiers and is git-ignored.

## Result in one sentence

The system cannot say "none of these", and its confidence only weakly signals when it should: confidence tells the two groups apart
with an area under the ROC curve of 0.62, where 0.50 is guessing.

## The comparison

| | In scope (2,315) | Other pathology (1,500) |
|---|---|---|
| names Normal | 32.7 percent | 38.1 percent |
| names Pneumonia | 60.4 percent | 52.4 percent |
| names Lung Cancer | 6.9 percent | 9.5 percent |
| answered without the low-confidence flag | 81.5 percent (79.5 to 83.3) | 72.4 percent (70.0 to 74.9) |
| prediction set has a single class | 56.3 percent (53.8 to 58.8) | 45.0 percent (42.2 to 47.7) |
| mean pooled top probability | 0.578 | 0.533 |
| cancer alert fires (in scope: studies that are not Lung Cancer) | 10.0 percent (8.0 to 12.2) | 18.2 percent (15.7 to 20.8) |
| AUROC of confidence as a detector of "outside the three classes" | | 0.616 (0.589 to 0.640) |

The pooled probability reached 0.85 for 1.0 percent of in-scope studies and 0.4 percent of the others; the pooling of three models
rarely produces probabilities that high (the calibrated confidence in the decision rules corrects for this), so that level says little
here and the low-confidence flag is the useful comparison.

## What it means

1. **Most studies of another disease get an ordinary answer.** Seven in ten were answered without a low-confidence flag, and
   half were named Pneumonia, which for effusion, edema or atelectasis may be a reasonable guess but is not a statement that the
   patient has pneumonia. Nearly four in ten were named Normal, a patient with an abnormal film and nothing the system can point to.
2. **The safeguards help a little, not enough.** The low-confidence flag fires on 27.6 percent of these studies against 18.5 percent of
   in-scope ones, and fewer have a single-class set (45 against 56 percent). Those are real but small differences. If the question is
   "can the confidence tell me this patient is outside what the system knows", the answer measured here is "barely".
3. **The cancer alert fires almost twice as often on other disease (18.2 percent) as on in-scope studies that are not cancer
   (10.0 percent).** Some non-cancer thoracic findings (masses of other kinds, effusions) look to the models like what they learned for
   cancer. Anyone reading an alert should know that.
4. **Dropping the Excluded studies had a cost that was not measured until now.** The models were never taught that "something else" exists, so
   their scores cannot express it.

## Limits

- An Excluded study has no correct answer among the three classes, so a confident answer is not "wrong" the way a wrong Pneumonia
  call on a Normal study is. This measures the missing ability to say "none of these", not an error rate.
- The Excluded set is whatever was left (mostly other findings on the report), not a designed set of other diseases. The sample contains
  only 2 tuberculosis-coded studies, too few to say anything about tuberculosis.
- The subgroup rates by report mention (effusion, edema, atelectasis, pneumothorax, cardiomegaly, fracture) are plain word matches
  with no handling of negation ("no pneumothorax" counts), so most studies match several terms. In the saved results they are all close to
  the overall rates and distinguish nothing, and are not quoted above.
- Patients not in training could still have been in the image backbone's pretraining (the caveat that the image-backbone
  experiment could not clear).
- One sample, one seed, validation-era data. The sealed test split was not used.

## What changes, and what does not

Nothing in the models. This is a documented limitation of version 1.5 and goes in the writeup with these numbers. A real fix is not a
patch to the decision layer: it needs the models to learn an "other" outcome (an explicit class, or an out-of-scope detector trained on
Excluded studies of training patients), which changes the models and would have to be decided before the sealed test evaluation, not
after. That was weighed against the plan to freeze the system for the test; it was not adopted here.
