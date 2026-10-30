# Version 2: teaching the models an "other" outcome. Plan, written before any model was trained

Written 2026-10-30. Nothing in this plan has been run. The sealed test split is spent for version 1, so this plan never opens it, and no number
from this work may be called a test number.

## Question

`experiments/other_pathology/RESULTS.md` showed that the system cannot say "none of these": on studies of other chest disease it answered
without a low-confidence flag 72 percent of the time, and its confidence told them apart from in-scope studies with an AUROC of 0.62.
`experiments/out_of_scope/RESULTS.md` then tried a gate built on the models' existing outputs and the image backbone's 18 outputs. It failed
(AUROC 0.573 on patients it had never seen), and its own diagnostic suggested the cross-validated 0.738 was an artifact of how the Excluded
predictions were made. The conclusion written there was that the information is not in the existing outputs, and that closing the gap needs an
explicit "other" outcome inside the models.

This plan tests that conclusion. If the three models are trained with a fourth class made of the Excluded studies, can the system tell, on
patients it has never seen, that a study is outside Normal, Pneumonia and Lung Cancer, without hurting the three classes it already handles?

## What "other" means, and what it does not

Other is every study the labelling rules set aside: no pneumonia or cancer code or report mention, and not a clean "No Finding" report with no
pneumonia, cancer or tuberculosis code (`scripts/exploration/build_final_labels.py`). That is a mixed bag: cardiac and pleural disease, other
infections, tuberculosis-coded admissions whose film may look normal, and anything else. It is not "abnormal" and it is not "healthy". The system
would say "possibly outside what I can assess", never name the other condition, and never advise anything. The page wording already exists
(`analyze.js`, the out-of-scope headline) and is reused.

## Built, and not built

Built: versions of the image, blood and text models with a fourth output, fused by the same geometric-mean rule, plus a flag derived from the
fused "other" probability. The flag never changes which of the three classes is the best guess, exactly like the low-confidence flag and the
cancer alert. A study that is not flagged gets the same kind of answer as today.

Not built here, and decided now so it cannot creep in later: naming which other disease it is, a fifth class, any change to the explainers,
the evidence chain or the page beyond the existing headline, retuning the version 1 models, or any use of the test split. If the experiment is
adopted, integration (rebuilding the decision rules for four classes, wiring the app, one confirmatory look at test data) gets its own plan.

## Data and the role of each part (patient-level, nobody is in two roles)

Counts are from the tables in `data/processed/`, checked on 2026-10-30.

| Studies | Patients | Role |
|---|---|---|
| in-scope train: 11,039 (Pneumonia 7,312, Normal 2,953, Lung Cancer 774) | 3,362 | training and out-of-fold predictions |
| Excluded studies of training patients: 5,369 | 1,408 | the fourth class in training |
| in-scope validation: 2,315 | 720 | negatives, looked at once |
| Excluded studies of validation patients: 1,234 | 336 | positives, looked at once |
| Excluded studies of patients in no split: 2,241 | 1,292 | positives for the final report, never used for any choice |
| in-scope test (2,099) and Excluded studies of test patients | 721 and more | not touched |

The split for the Excluded studies follows the patient's existing split. A patient with in-scope and Excluded studies keeps all of them in the
same role. The Excluded tables come from `scripts/build_final_dataset.py --excluded --role train|val|none`. Test patients cannot be named by that
script, which is the safeguard.

The Excluded studies are a large share of the linked data (at least 8,844 against 15,453 in scope), but any real clinic's mix is unknown. So
this work reports AUROC, the share of Excluded studies flagged, and the share of in-scope studies flagged. It does not report a positive
predictive value as if it were a deployment number.

## Step 0, before any training

Checks that can still change the plan, run and written down first:

1. The cached image backbone outputs exist for all in-scope training and validation studies (the Excluded ones are in
   `experiments/out_of_scope/feats_*.npz`). If they exist, the image head trains locally in minutes. If not, they are extracted locally on the CPU
   (about an hour, resumable, as in `scripts/extract_image_backbone_features.py`). No Kaggle run is planned.
2. The share of Excluded studies whose report says "No Finding" (the tuberculosis-coded normals). They are Other by label and Normal-looking by
   film, so they cap what any image or text model can do. The number goes in the results.
3. Hashes of the version 1 artifacts, so the version 1 system provably did not change.

## Folds

The version 1 out-of-fold recipe is reused exactly: 5-fold `StratifiedGroupKFold` grouped by patient, seed 42. The Excluded studies join the
fold of their patient. (Step 0 found that every patient with Excluded training studies also has in-scope training studies, so no patient needs a fold
of their own.) The in-scope assignment will not be identical to version 1, because the splitter stratifies over four classes now. That is stated, and the
version 1 training-time out-of-fold scores are recomputed on the new folds as the comparison, so nothing is compared across different folds.

## Arms (fixed now)

| Arm | What is trained | Where "other" comes from |
|---|---|---|
| J, joint | each modality gets a fourth output and is retrained on four classes with the version 1 recipe and hyperparameters, `class_weight="balanced"`, no tuning | the fused fourth probability |
| T, two-stage | the version 1 three-class models are unchanged; one extra binary "Excluded versus in-scope" model per modality on the same features, same recipe | the geometric mean of the three binary scores |

Fusion is the geometric mean in both, as in version 1. Arm T is the cautious design: the three-class answers are identical to version 1 by
construction. Arm J is the one the version 1 conclusion points at. The arm is chosen on the training out-of-fold predictions only. Higher
cross-validated AUROC of the "other" score wins, and a tie within 0.005 goes to T, because it leaves the audited system untouched. Not tried:
other model families, a text or image encoder change, focal loss, a threshold sweep over many settings. Adding any of them after seeing results
would be fishing.

The image head is a small network on 18 numbers, so it is trained with 3 seeds and the averaged probabilities are used. The seed spread is
reported, because retraining the same image recipe earlier moved the image macro F1 by about 0.03.

## Operating point

Flag a study as "possibly outside" when the fused other probability reaches a threshold. The threshold is the one that flags 10 percent of the
in-scope training studies by their out-of-fold scores. How many false flags are acceptable is a product decision, as it was for the cancer alert,
so the results also show the flag rate at other settings (5 and 20 percent).

For a study that is not flagged, and for the underlying best guess, the class is the best of the three in-scope classes, renormalised. So the
three-class prediction rule is the same in both arms and in version 1.

## Adoption rule (to be encoded in the training script as a function `adopt`, written before the first run)

Measured on the in-scope validation studies (negatives) and the 2,241 Excluded studies of patients in no split (positives), with 95 percent
intervals from a bootstrap that resamples patients (2,000 draws). The version is adopted only if all hold:

1. The AUROC of the other score is at least 0.75 and its lower bound is at least 0.70. The existing confidence signal scored 0.62 and the failed
   gate 0.573.
2. At the operating point, at least 35 percent of the held-out Excluded studies are flagged and at most 13 percent of in-scope validation studies
   are (the target is 10).
3. The three-class macro F1 over all in-scope validation studies, ignoring the flag, is at least 0.583. Version 1 scored 0.613, and 0.03 is the
   measured retraining noise of the image head, so this is "not worse by more than noise".
4. At most 20 percent of true Lung Cancer validation studies are flagged.
5. After temperature scaling fitted on out-of-fold training predictions, the expected calibration error of the three-class probabilities on
   in-scope validation is at most 0.05. Version 1 reached 0.028, and every downstream rule (low confidence, prediction sets, cancer alert) depends
   on calibration.

The Excluded studies of validation patients (1,234) are scored once as a second set of positives. They are reported and are not part of the
rule, because the final report should rest on patients no choice ever touched.

If any condition fails, version 1 stays as it is and this is reported as a negative result, with the same weight as a positive one.

## Reported whatever the outcome

Per-class precision, recall and F1 on in-scope validation for the chosen arm beside version 1. The share of "No Finding" Excluded studies flagged
against the rest. The flag rate by modality availability (all three inputs, no labs, no indication). The seed spread of the image head. A paired
comparison with the failed gate on the same held-out patients.

## Limits stated in advance

- The Excluded class is whatever the rules left over, not a designed set of diseases, and its label does not say which condition it is. A study of
  a disease that is rare here may behave nothing like the average Excluded study.
- The text model sees only the reason for the exam. (An earlier draft also listed tuberculosis-coded normals as a cap on the class. Step 0 measured
  them at 0.1 percent of Excluded studies, so that limit was wrong; see the step 0 results below.)
- Excluded studies of training patients get predictions from models that saw other studies of the same patients. The out-of-fold scheme and the
  patient-level roles exist to stop that from inflating anything. The failed gate fell exactly there, so the held-out patients are the only
  numbers to quote.
- One hospital, one labelling pipeline. A detector trained here says nothing about disease at another site.
- The in-scope test split is spent, so the cost on in-scope cases is measured on validation only, and the in-scope validation studies were used in
  version 1 choices. Their use here is limited to one look, which is the best that can be done without new data.

## Order of work, one step at a time with a report after each

1. Step 0 checks above, and a test that fails if any tracked file or output names the test split's patients.
2. Build the four-class tables and folds. Tests: no patient in two roles, folds group by patient, class counts as in the table.
3. Arm T and arm J out-of-fold predictions for blood, text and image, and the choice of arm on training data only.
4. Fusion, operating point, calibration, and the `adopt` function with tests that break each condition on purpose.
5. One run on validation and the held-out patients, then `RESULTS.md`, written whatever the outcome, plus a dashboard card.
6. Only if adopted: a separate plan for integration and for one clearly labelled post-hoc look at test data.

## Cost

Everything runs locally on the CPU: blood and text models are the version 1 recipes on more rows, the image head trains on cached 18-number
inputs, and the text embeddings are cached. Zero cost, no GPU, no Kaggle. The one slow step is image feature extraction if step 0 finds the
cached outputs missing.

## Step 0 results (2026-10-30, `scripts/v2_step0.py`, numbers in `step0.json`)

Run before any training. Two statements in the text above were corrected by it and nothing else was changed.

1. Cached image outputs are usable. In-scope training (11,039 rows) and validation (2,315) match their tables in length, and the first 6 films of each,
   recomputed from the image, agree with the cache to 1e-6. The Excluded outputs carry study ids and are in table order for all three roles.
   So no image extraction is needed, on any machine.
2. The Excluded class is almost entirely studies whose report states a finding other than pneumonia or cancer: the share whose report says
   "No Finding" is 0.1 percent (train), 0.1 (validation) and 0.0 (no split), and the share with a tuberculosis code is 0.4, 0.2 and 0.0 percent.
   My draft expected a sizeable group of normal-looking films coded as tuberculosis. There is none. That limit is removed, and the class is better
   described as "reported abnormal, but not as pneumonia or cancer". Two consequences. The class is label-defined by what the radiologist wrote, so a
   film-based signal is plausible. And a study the reports call pneumonia or cancer in passing is not Other, so the boundary to the two disease
   classes is blurry, which will show up as Other flags on Pneumonia studies.
3. Availability of inputs in the Excluded studies is close to the in-scope pattern: labs present in 98.9, 98.5 and 94.4 percent, an indication in 96.1,
   96.6 and 96.1 percent (train, validation, no split). The no-split group has the fewest labs, so its flag rate should be read with that in mind.
4. The slow step is slower than I guessed. Embedding 128 Excluded indications with Bio_ClinicalBERT took 31 s on the CPU (10 threads, no GPU),
   0.24 s per text, so all 8,844 would take about 36 minutes, not the 10 I estimated. That is still cheap. Batches will be sorted by length before
   embedding, because the median indication is 8 words and a few are over 100, and padding to the longest in each batch of 32 wastes most of the work;
   sorting changes the speed and not the result. The run is chunked and resumable. A GPU or Kaggle was considered and not used, for the reasons given
   when the plan was approved.
5. Hashes of the 12 frozen version 1 files, the backbone weights, the git commit and 12 data and prediction files are in `step0.json`, so the end
   report can state that version 1 did not change.

6. A structural finding that the first draft of this plan missed. The "no split" patients (2,241 Excluded studies, 1,292 patients) are, by how the
   tables are built, exactly the patients who have no in-scope study at all, in any split. Every Excluded study in the training and validation roles
   belongs to a patient who also has in-scope studies (a test now checks this). So the held-out positives are patients whose every study fell
   outside the three classes, while the training positives are the other studies of patients who also had Normal, Pneumonia or cancer studies. These two
   groups need not look alike: the held-out group has fewer labs on record (94.4 against 98.9 percent). A score that works on one may not work on the
   other, and this probably contributed to the failed gate's gap between its cross-validated and held-out numbers. This is a real limit of the evidence
   and not something to fix by moving patients between roles after seeing results. The plan handles it by reporting the validation Excluded studies
   (training-like patients, 1,234) and the held-out ones (2,241) side by side. If they disagree, that disagreement is the finding.

The adoption rule, arms, roles and limits above are unchanged.
