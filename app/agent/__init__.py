"""
Phase 5 agentic layer.

The design rule this package exists to enforce: the agent orchestrates and reasons, the
trained models diagnose, and nothing here ever invents a prediction. Every probability
that reaches the belief state comes from explain.image / explain.blood / explain.text,
which are the same saved Phase 2 models used throughout the project.

    tools.py         thin, uniform wrappers around the three explain.* modules
    belief.py        pools whichever modalities have run, using the exact rule
                      validated in Phase 3 (scripts/train_fusion.py, logpool)
    consistency.py   flags when the modalities disagree, using the real accuracy-by-
                      agreement numbers from the validation set, not an invented rule
    planner.py       decides whether more input is worth requesting, using the real
                      modality ablation ranking from Phase 3
    orchestrator.py  runs the above in sequence for one case
"""
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]
