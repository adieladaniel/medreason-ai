# Feature-level (intermediate) fusion, MLP ablation

Train fit/dev/val: 9310/1729/2315. Dev is used only to choose the text PCA dimension and to early-stop training; val is touched once, for the number below.

Text PCA dimension scan (dev macro F1): {'16': 0.6241, '32': 0.6597, '64': 0.6562}. Chosen: 32.

## Result (val)
Macro F1 **0.5864** (Normal 0.6206 / Pneumonia 0.7591 / Cancer 0.3794), 88 input features.

**0.5864 vs late fusion (logpool) 0.613 -> late fusion still wins**

## Modality-subset ablation (macro F1)
| modalities | dev | val |
|---|---|---|
| image | 0.5125 | 0.4432 |
| blood | 0.4575 | 0.4171 |
| text | 0.5957 | 0.5286 |
| image+blood | 0.5315 | 0.4895 |
| image+text | 0.6456 | 0.5578 |
| blood+text | 0.6261 | 0.5691 |
| image+blood+text | 0.6554 | 0.5864 |

## Comparison to late fusion (scripts/train_fusion.py)
Late fusion (logpool): 0.613. Feature fusion (this): 0.5864.

Artifacts: fusion_mlp.pth, val_predictions.csv, metrics.json