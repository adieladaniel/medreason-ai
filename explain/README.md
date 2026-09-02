# `explain/` — Phase 4 explainability

One module per modality. Each loads a **saved** Phase 2/3 model (never imports training
code) and turns one case's prediction into an attribution that a human and the Phase 5
agent can both read.

## Common output contract

```python
{
  "modality": "blood" | "text" | "image",
  "predicted_class": "Normal" | "Pneumonia" | "Lung Cancer",
  "probabilities": {class: float},
  "attributions": {
      "method": str,
      "items": [{"feature": str, "value": float|None, "effect": float, "direction": "supports"|"against"}, ...]
      #  effect is signed, w.r.t. the predicted class; items sorted by |effect|
  },
  "summary": str   # one-line natural-language gloss
}
```

## Modules

| module | model explained | method | notes |
|---|---|---|---|
| `blood.py` | `HistGradientBoostingClassifier` | SHAP `TreeExplainer`, tree-path-dependent (exact) | contributions in log-odds; `softmax(base + Σshap) == predict_proba` |
| `text.py` | frozen Bio_ClinicalBERT (mean-pooled) → LogisticRegression, INDICATION only | SHAP with a word-level (`\W+`) `Text` masker over the whole embed→classify pipeline | ~60–90 s/case CPU (`max_evals=300`); per-word effect on P(predicted) |
| `image.py` | frozen torchxrayvision DenseNet121 → 18 pathology probs → MLP head | Grad-CAM (last conv layer, predicted-class logit) **+** pathology-channel attribution `d(logit)/d(prob)·prob` | input needs `requires_grad_(True)` — the backbone is frozen |

## Entrypoint

```bash
python -m explain.run --study-id 53383543 --save experiments/explain_samples/
python -m explain.run --study-id 53383543 --only blood,text
python -m explain.blood --study-id 53383543 --plot out.png     # single modality
python -m explain.text  --text "65M smoker, chronic cough, weight loss"
```

`explain/run.py` runs all three for a `study_id` (from `val.csv`), prints a combined
view, and writes per-modality PNGs + a JSON. **Its output goes to
`experiments/explain_samples/`, which is git-ignored** (contains X-ray images and
indication text).

## Gotchas

- `import torch` must run before any `sklearn` / `shap` import in the process, or torch's
  DLL init fails on Windows (`WinError 1114`). The modules do this at the top; if you
  write a new caller, do the same.
- torchxrayvision's weight downloader prints a `█` progress bar that crashes on Windows
  cp1252. Pre-fetch the `.pt` into `~/.torchxrayvision/models_data/` with curl, or run
  with `PYTHONIOENCODING=utf-8`.
- The text explainer is too slow for a live UI — pre-compute and cache explanations for
  the demo cases.
