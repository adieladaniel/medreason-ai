"""
Phase 4 explainability -- one module per modality, each loading a saved Phase 2/3 model
and turning a single case's prediction into a human- and agent-readable attribution.

Not entangled with training code (CLAUDE.md rule): these modules only read saved models
(experiments/*/), plus train.csv where an explainer needs a background/reference sample.

Common output contract (dict):
    modality:          "blood" | "text" | "image"
    predicted_class:   "Normal" | "Pneumonia" | "Lung Cancer"
    probabilities:     {class: float}
    attributions:      {method: str, items: [{feature/token, value, effect, direction}, ...]}
                       -- effect is signed, w.r.t. the predicted class; items sorted by |effect|
    summary:           one-line natural-language gloss
"""
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]
