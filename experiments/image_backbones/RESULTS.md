# Image backbone experiment: results

Run on 2026-10-29 with `scripts/extract_image_backbone_features.py` and `scripts/eval_image_backbones.py`, following `PLAN.md`
(written before any feature was extracted). Validation split only; the sealed test split was not opened. Numbers are in
`results.json`.

## Outcome

**The current image model stays.** No arm met the pre-registered adoption rule. This is a negative result, and the rule fixed in
advance is what decided it.

## What was compared

The same small head (inputs, 64 units, 3 outputs, 15 epochs, seed 42, the same out-of-fold folds) on four sets of frozen outputs.
Image alone, then fused with the stored blood and text models by the app's geometric-mean pool. 95 percent intervals resample
patients; differences are paired against the control.

| Arm | Image alone: Lung Cancer AUC | Image alone: Pneumonia F1 | Fused macro F1 | Fused Lung Cancer AUC |
|---|---|---|---|---|
| control (`mimic_ch`, current) | 0.534 (0.447 to 0.602) | 0.714 | 0.613 (0.569 to 0.649) | 0.855 (0.797 to 0.899) |
| nih (14 trained outputs incl. Nodule, Mass) | 0.515 (0.447 to 0.587) | 0.604 | 0.600 (0.556 to 0.635) | 0.857 (0.801 to 0.901) |
| pc (15 trained outputs incl. Nodule, Mass) | 0.604 (0.515 to 0.673) | 0.676 | 0.602 (0.558 to 0.638) | 0.861 (0.805 to 0.904) |
| nih + mimic_ch | 0.530 (0.457 to 0.598) | 0.668 | 0.616 (0.570 to 0.652) | 0.856 (0.799 to 0.900) |

Paired difference from the control (point, 95 percent interval):

| Arm | Image Lung Cancer AUC | Image Pneumonia F1 | Fused macro F1 | Fused Lung Cancer AUC |
|---|---|---|---|---|
| nih | -0.019 (-0.109 to +0.087) | -0.110 (-0.141 to -0.079) | -0.013 (-0.027 to +0.001) | +0.003 (-0.004 to +0.010) |
| pc | +0.070 (-0.003 to +0.144) | -0.038 (-0.075 to -0.002) | -0.011 (-0.023 to +0.001) | +0.006 (-0.001 to +0.014) |
| nih + mimic_ch | -0.004 (-0.063 to +0.065) | -0.046 (-0.063 to -0.029) | +0.003 (-0.006 to +0.013) | +0.001 (-0.002 to +0.005) |

Why each arm failed the rule: nih and nih + mimic_ch show no gain in image Lung Cancer AUC; pc has the best AUC (+0.070) but its
paired interval reaches -0.003, so it is not clearly above the control, and it lowers the fused macro F1 (-0.011).

## What it means

1. **Swapping in Nodule and Mass outputs did not fix the image model's blindness to cancer.** The NIH weights, which have trained
   Nodule and Mass channels, scored no better than the MIMIC weights, which do not (0.515 against 0.534). The most likely reason is the
   one already suspected and not provable here: the Lung Cancer label comes from admission-level codes and report mentions, so for many
   studies the film itself may show nothing to find. A frozen backbone's output probabilities are also a narrow window; embeddings
   or fine-tuning were left out of this experiment on purpose and may behave differently.
2. **PadChest weights are a hint, not a result.** An image Lung Cancer AUC of 0.604 with a paired interval of -0.003 to +0.144 is the
   kind of number that could be real or noise, and it did not carry into the fused result (fused Lung Cancer AUC +0.006, fused macro
   F1 -0.011). Three arms were compared on one validation set, so the best-looking one is somewhat flattered.
3. **Leakage control: inconclusive, leaning towards a domain-shift story.** The control's image Pneumonia F1 (0.714 here, 0.732 in the
   stored predictions) is higher than every non-MIMIC backbone's: nih 0.604 (-0.110, interval -0.141 to -0.079), pc 0.676 (-0.038,
   interval -0.075 to -0.002). The plan reads the nih drop as "consistent with inflation, but a domain shift is a competing explanation"
   and the pc drop as "inconclusive". Because both explanations predict the same drop, this experiment cannot say whether the
   MIMIC-pretrained backbone saw our patients. The writeup must keep that caveat, not claim it is cleared.
4. **Retraining noise is about 0.03.** The same recipe retrained here scored image-alone macro F1 0.472 against 0.481 in the stored
   predictions (the 0.50 in the notes is the earlier Phase 2 run with augmentation), and Pneumonia F1 0.714 against 0.732. Any
   image-alone difference below roughly 0.03 should not be read as a difference between backbones.

## Gate 0

Passed: the retrained control gave fused macro F1 0.6129 (stored 0.6133) and image-alone macro F1 0.472. The plan compared the
latter with 0.50, which is the Phase 2 number; the stored out-of-fold kernel predictions that fusion uses score 0.481. The gate
passed against either reference, and the plan's reference was wrong, not the result.

## Limits

- Frozen outputs only, one head recipe, one seed. Embeddings, fine-tuning, other head sizes and other seeds were not tried.
- Three comparisons against the control on the same validation set; 117 Lung Cancer studies make every cancer number wide.
- The 18 outputs of each backbone are probabilities from a model trained on another hospital's labels, then read by our head. They are
  not the same thing as features learned for our labels.
- Provenance of the weights: downloaded from the torchxrayvision release page on 2026-10-29; SHA-256 prefixes
  `23b13a04` (mimic_ch), `31a5475c` (nih), `a9148ef6` (pc).

## What changes in the project

Nothing in the models. The notes' open item "try image weights that have Nodule and Mass channels" is closed by this result, and the
image input remains the weakest of the three for Lung Cancer, as already reported.
