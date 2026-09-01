# Phase 3 -- late multimodal fusion (results)

Fusion train n=11039, val n=2315. Meta-model input = 9 probabilities (image|blood|text, 3 classes each). Text = Setting A (indication). Metric: val macro F1 (test sealed).

## Single modality (val, from the same prediction files)
| modality | macro F1 | Normal | Pneumonia | Cancer |
|---|---|---|---|---|
| image | 0.4807 | 0.5609 | 0.732 | 0.1491 |
| blood | 0.4609 | 0.453 | 0.7296 | 0.2 |
| text | 0.5094 | 0.5251 | 0.6868 | 0.3163 |

## Fusion
| method | macro F1 | Normal | Pneumonia | Cancer |
|---|---|---|---|---|
| mean_probability | 0.5996 | 0.6098 | 0.7863 | 0.4026 |
| logpool | 0.6133 | 0.6175 | 0.7974 | 0.4249 |
| logreg_stack_balanced | 0.5623 | 0.6162 | 0.7291 | 0.3415 |
| logreg_stack | 0.5848 | 0.5692 | 0.8328 | 0.3523 |
| hgb_stack | 0.5605 | 0.6121 | 0.7382 | 0.3311 |

**Ablation (mean rule, val macro F1):** {'image': 0.4807, 'blood': 0.4609, 'text': 0.5094, 'image+blood': 0.4925, 'image+text': 0.557, 'blood+text': 0.5762, 'image+blood+text': 0.5996}

**Verdict:** 0.6133 vs best single 0.5094 (text) -> fusion HELPS

## Notes
- Every class improves over its best single-modality F1 (Normal, Pneumonia, and especially **Cancer**, the class that was weak everywhere in Phase 2).
- The parameter-free pooling rules (logpool / mean) beat the trained stackers (logreg / hgb). With only 9 inputs and reasonably calibrated base probabilities, a trained meta-model mostly overfits noise or over-corrects toward Cancer recall; the log-opinion pool is the principled combiner for independent probabilistic classifiers and wins here empirically too.
- Ablation: all three modalities beat every pair; text contributes most, image least (image+blood is the weakest pair).
- Leakage guard: text inputs are Setting A (INDICATION section only).