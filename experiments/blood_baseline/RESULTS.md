# Blood/tabular unimodal baseline -- results

Train n=11039, Val n=2315. Metric: macro F1 on val (test split untouched).

| model | macro F1 | balanced acc | Normal F1 | Pneumonia F1 | Cancer F1 |
|---|---|---|---|---|---|
| dummy_most_frequent | 0.2657 | 0.3333 | 0.0 | 0.7971 | 0.0 |
| logreg | 0.4002 | 0.4971 | 0.3852 | 0.6249 | 0.1906 |
| hist_gradient_boosting | 0.4609 | 0.4688 | 0.453 | 0.7296 | 0.2 |

HGB macro F1 on the 2248 val rows with >=1 core lab: 0.4558

Top features (permutation importance): hemoglobin, albumin, ldh, wbc, lymphocytes, urea

Artifacts: hgb_model.joblib, logreg_model.joblib, feature_importance.csv, val_predictions.csv (blood-model probabilities for Phase 3 fusion), metrics.json

## Comparison to the image baseline (v3, val)
Image v3: macro F1 0.50 (Normal 0.56 / Pneumonia 0.77 / Cancer 0.17).