# Image backbone experiment: plan, written before any feature was extracted

Written 2026-10-29. Validation split only; the test split is never loaded by this experiment.

## Question

The image input adds the least to the fused result (+0.026 macro F1) and cannot be told from guessing on Lung Cancer (ROC AUC
0.527, 95 percent interval 0.440 to 0.600). The current backbone (`densenet121-res224-mimic_ch`) has no trained Nodule or Mass
output (7 of its 18 outputs are the constant 0.5), and the notes carry an unverified caveat that it was pretrained on MIMIC-CXR
images that may include our validation patients.

Does a backbone that was trained to output Nodule and Mass, and that has no known overlap with MIMIC-CXR, make the image input
useful for Lung Cancer, and does that survive fusion?

## Arms (fixed now)

Everything except the frozen backbone is identical to the current image model: the preprocessing in `explain/image.py`, the head
(inputs, 64 ReLU units, dropout 0.2, 3 outputs), 15 epochs, Adam 1e-3, cosine schedule, class-weighted loss, seed 42, no
augmentation, and the same out-of-fold folds (5-fold StratifiedGroupKFold by subject, seed 42) as the other modalities.

| Arm | Backbone weights | Trained outputs | Pretraining data |
|---|---|---|---|
| control | `mimic_ch` | 11 | MIMIC-CXR |
| nih | `nih` | 14 including Nodule and Mass | NIH ChestX-ray14 |
| pc | `pc` | 15 including Nodule and Mass | PadChest |
| nih+mimic | `nih` and `mimic_ch`, outputs joined | 25 distinct | both |

Not included, on purpose: the `all` weights (their training data includes MIMIC-CXR, so they have the same overlap question),
penultimate-layer embeddings, fine-tuning, other head sizes. Adding any of them after seeing results would be fishing.

## Measurements (validation, 2,315 studies, 117 Lung Cancer)

For each arm: image-alone macro F1, per-class F1 and per-class ROC AUC; and the fused result, the geometric-mean pool of the arm's
image probabilities with the stored blood model and the stored text model (indication only), as in `scripts/train_fusion.py`:
macro F1, Lung Cancer F1, Lung Cancer ROC AUC. Intervals are 95 percent cluster bootstrap over subjects (2,000 draws), and
differences from the control are paired (same resampled subjects).

## Gate 0, before reading any other arm

The control, re-extracted and retrained here on the CPU, must reproduce the stored one: fused macro F1 within 0.015 of 0.6133,
and the image-alone macro F1 within 0.03 of 0.50. If it does not, stop and find out why before looking at the other arms.

## Adoption rule (encoded in `scripts/eval_image_backbones.py`, function `adopt`)

An arm replaces the production image model only if all of these hold:

1. Image-alone Lung Cancer AUC is above the control's with a paired 95 percent interval whose lower end is above 0, and the arm's
   own interval lower end is above 0.50.
2. Fused macro F1 is not lower than the control's (paired difference at least 0) and the lower end of its paired interval is at
   least -0.02.
3. Fused Lung Cancer AUC is not lower than the control's (paired difference at least 0).

If several arms pass, take the one with the highest fused macro F1; if two are within 0.005, take the one that needs fewer
backbones. If none passes, the current image model stays and this is reported as a negative result.

With three arms tried against the control on the same validation set, the winner's numbers are somewhat optimistic. The sealed test
split is what corrects that, and it is used once, later, with whatever model is chosen.

## Leakage-control readout (reported whatever happens)

The current image model's best score is Pneumonia F1 0.77. If the `nih` and `pc` arms (no MIMIC pretraining) score within 0.03 of
that, there is no evidence that overlap with the backbone's pretraining inflated it. If they score more than 0.05 lower, that is
consistent with inflation, but not proof, because the shift from NIH or PadChest images to MIMIC images is a competing
explanation and the two cannot be separated here. In between, the answer is inconclusive. This readout does not affect adoption.

## Stop rules

Two working days. If feature extraction is far slower than measured below, extract the arms one at a time and report what was
finished.

## If an arm is adopted (separate work, not part of this experiment)

New out-of-fold and validation prediction files, then in order: fusion (`train_fusion.py`), decision rules
(`build_decision_rules.py`), the reliability analysis (`eval_uncertainty.py`), the dashboard data, `explain/image.py` and
`app/prefetch.py`, the tests and docs that quote the old numbers, and the end-to-end rerun.

## Amendment 1 (2026-10-29, before any arm was run)

Gate 0 said the control would be re-extracted. A check on 64 validation images showed the local CPU extraction with the
`mimic_ch` weights equals the cached Kaggle features to within 0.000002 (largest absolute difference), so the control and the
control half of the nih+mimic arm use the cached `experiments/image_baseline/feats_{train,val}.npy`. Re-extracting would cost
about 25 minutes and give the same numbers. The control head is still retrained here, so the rest of Gate 0 stands: it must
reproduce fused macro F1 0.6133 within 0.015 and image-alone macro F1 0.50 within 0.03.

Measured before the run: the local CPU extracts 6 to 11 images per second per backbone on this loaded laptop, so the new
backbones (13,354 train and validation images) take about an hour in one pass.
