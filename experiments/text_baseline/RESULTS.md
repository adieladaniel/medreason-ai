# Text unimodal baseline -- results

Encoder: frozen emilyalsentzer/Bio_ClinicalBERT, mean-pooled. Head: multinomial logistic regression (class_weight balanced). Metric: macro F1 on val (test untouched).

## Setting A -- `indication` section only  (THE honest number)

| head | macro F1 | Normal F1 | Pneumonia F1 | Cancer F1 |
|---|---|---|---|---|
| logreg | 0.5094 | 0.5251 | 0.6868 | 0.3163 |
| mlp | 0.51 | 0.4543 | 0.7846 | 0.2911 |

## Setting B -- full report  (LEAKAGE-INFLATED, not a capability estimate)

logreg macro F1 0.675 (Normal 0.774 / Pneumonia 0.8341 / Cancer 0.4169). Labels were partly derived from this text; the score reflects that.

## Leakage analysis -- macro F1 split by how the label was derived

| val subset | n | Setting A (indication) | Setting B (full report) |
|---|---|---|---|
| label had report-derived evidence | 1758 | 0.5497 | 0.7457 |
| label is ICD-only (report independent) | 557 | 0.3002 | 0.3347 |

Reading: on the ICD-only subset (labels the report cannot leak), seeing the whole report barely beats seeing only the indication (0.3347 vs 0.3002) -- so Setting B's headline lead over Setting A (0.675 vs 0.5094) is almost entirely the report-derived subset (0.7457 vs 0.5497), i.e. leakage. Setting A is close to the true text ceiling. (The report vs ICD-only gap within Setting A also partly reflects task difficulty + class mix: the ICD-only subset is all Pneumonia/Cancer with no clear radiographic finding, and contains no Normal cases.)

## Baseline comparison (val macro F1)
- Image v3 (frozen MIMIC-CXR DenseNet): 0.50  -- strongest on Pneumonia (F1 0.77)
- Blood (HistGradientBoosting): 0.46
- Text A (indication): 0.5094  -- strongest on Lung Cancer (F1 0.3163, vs image 0.17 / blood 0.20)
The three modalities are complementary -> Phase 3 fusion has a real case.

Artifacts: logreg_indication.joblib, logreg_full_report.joblib, val_predictions.csv (both settings' probabilities, for Phase 3 fusion), cached embeddings emb_*.npy, metrics.json