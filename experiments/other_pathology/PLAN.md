# What does the system do with disease it has no class for? Plan, written before the data was built

Written 2026-10-29. Validation-era data only; the sealed test split is not touched.

## Question

The system answers Normal, Pneumonia or Lung Cancer. When the dataset was built, every study that was none of these (some other
chest pathology, or a tuberculosis admission with a clean report) was labelled Excluded and dropped, 10,049 of 25,502 linked
studies. So the models have never been shown "something else". In use, such a patient gets pushed into one of the three classes.
The low-confidence flag and the prediction set are meant to soften that, but nobody has measured whether they do.

How does the unchanged system behave on studies whose condition is outside its three classes, compared with studies inside them?

## Data

The Excluded studies of patients who are in neither the training split nor the sealed test split (`scripts/build_final_dataset.py
--excluded`, which keeps a patient only if the system has never been trained on them). Each gets the same inputs the app would get:
X-ray, labs when on record, the indication from the report. Up to 1,500 are drawn with seed 42.

The comparison group is the 2,315 validation studies (in scope), scored by the same pooling and decision code from the stored
per-modality probabilities, which the end-to-end run showed match the live models to 0.0001.

## What is measured (no pass or fail; this is a measurement)

The Excluded studies go through the app's own prediction path (`app/pipelines/staged.stage_predict`), so the pooling, the
calibrated low-confidence rule, the prediction set and the cancer alert are the production ones. For both groups:

1. Which class the system names (Normal, Pneumonia, Lung Cancer).
2. How often it sounds sure: not flagged low-confidence; a pooled probability of at least 0.85 (the planner's "high" level); a
   prediction set with a single class.
3. How often the cancer alert fires. For the in-scope group this uses only studies that are not Lung Cancer, so it is a false-alarm
   rate on both sides.
4. Whether confidence tells the groups apart: the area under the ROC curve for the pooled maximum probability as a detector of
   "outside the three classes" (0.5 means it cannot tell, 1.0 means it always can), and how many Excluded studies the existing
   low-confidence flag catches at the rate it flags in-scope studies.
5. Descriptively, for two kinds of Excluded study: tuberculosis-coded admissions, and reports that mention effusion, edema,
   atelectasis, pneumothorax, cardiomegaly or a fracture. The mentions are plain word matches with no negation handling, so they are a
   rough description, not labels.

Intervals are 95 percent cluster bootstrap over patients, 2,000 draws.

## What the result can and cannot say

- An Excluded study has no "correct" answer among the three classes, so a confident answer on one is not wrong in the sense that a
  wrong Pneumonia call on a Normal study is. It shows the system cannot say "none of these", and how well it signals that.
- Some Excluded pathologies (effusion, atelectasis, edema) look like pneumonia on a film, and the model may be partly right to say
  Pneumonia. The measurement does not separate "reasonable" from "overconfident".
- The Excluded set is a mixture of whatever was left, not a designed set of other diseases. Subgroup counts are small for the rare kinds.
- Patients not in training may still have been in the image backbone's pretraining (the open caveat from the image experiment).

## After the measurement

If confidence barely separates the groups, the honest outcome is a stated limitation, not a quick patch. Candidate mitigations (an
explicit "other" class or an out-of-scope detector trained on Excluded studies from training patients) are separate work that would
change the models, so they would have to be decided before the sealed test evaluation, not after.
