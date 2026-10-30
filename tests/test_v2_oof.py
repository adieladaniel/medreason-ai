"""Version 2, step 2 (scripts/v2_oof.py): the arm choice rule and the 'other' score of each arm. The model fitting needs the data and takes
minutes, so it is exercised by the run itself and by the cross-check against the stored version 1 predictions it records."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("v2_oof", ROOT / "scripts" / "v2_oof.py")
oof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oof)


@pytest.mark.parametrize("j, t, chosen", [(0.80, 0.70, "J"), (0.7551, 0.75, "J"), (0.7549, 0.75, "T"), (0.75, 0.75, "T"), (0.70, 0.80, "T")])
def test_arm_j_wins_only_by_more_than_the_tie_margin(j, t, chosen):
    assert oof.choose(j, t) == chosen


def test_the_tie_margin_is_the_one_in_the_plan():
    assert oof.TIE == 0.005
    assert "within 0.005" in (ROOT / "experiments" / "v2_four_class" / "PLAN.md").read_text(encoding="utf-8")


def test_each_arm_reads_its_own_column_of_the_pooled_vector():
    rng = np.random.default_rng(0)
    four = {m: rng.dirichlet([1, 1, 1, 1], 30) for m in ("blood", "text", "image")}
    two = {m: rng.dirichlet([1, 1], 30) for m in ("blood", "text", "image")}
    j, t = oof.other_score("J", four), oof.other_score("TB", two)
    assert j.shape == t.shape == (30,) and ((0 < j) & (j < 1)).all() and ((0 < t) & (t < 1)).all()
    certain = {m: np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)) for m in four}
    assert oof.other_score("J", certain).min() > 0.99
    certain2 = {m: np.tile([0.0, 1.0], (5, 1)) for m in two}
    assert oof.other_score("TB", certain2).min() > 0.99


def test_no_scoring_of_validation_or_heldout_rows_happens_in_the_training_script():
    code = (ROOT / "scripts" / "v2_oof.py").read_text(encoding="utf-8")
    assert "test.csv" not in code
    main_part = code[code.index("def main"):]
    for forbidden in ("roc_auc_score(labels", "f1_score(lib.labels(tables[\"val\"", "tables[\"heldout\"]"):
        assert forbidden not in main_part
