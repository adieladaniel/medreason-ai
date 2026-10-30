"""
Step 1 of the version 2 plan (experiments/v2_four_class/PLAN.md): the four-class tables and the patient-grouped folds.

Roles (patient-level, nobody is in two roles):
  train    in-scope training studies plus the Excluded studies of training patients, labelled Other, with a cross-validation fold
  val      in-scope validation studies plus the Excluded studies of validation patients, labelled Other
  heldout  the Excluded studies of patients who are in no split. Never used for any choice; the final report only.
The sealed test split is not read here: this script has no input for it, and tests/test_v2_step0.py and tests/test_v2_tables.py check that
no patient in these tables is a test patient.

Folds repeat the version 1 recipe (5-fold StratifiedGroupKFold, grouped by patient, seed 42) over four classes instead of three. The in-scope
rows therefore do not all keep their version 1 fold, and the fraction that do is reported, because the version 1 comparison scores are recomputed
on the new folds rather than compared across different ones.

Writes data/processed/v2_{train,val,heldout}.csv (git-ignored, they hold study ids and paths) and experiments/v2_four_class/tables.json
(counts only). Run from the repo root:  python scripts/v2_build_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "v2_four_class"

CLASSES = ["Normal", "Pneumonia", "Lung Cancer", "Other"]
N_FOLDS = 5
SEED = 42


def label_other(excluded: pd.DataFrame) -> pd.DataFrame:
    """The Excluded studies become the fourth class. Anything else in that table is a mistake, so it stops the build."""
    bad = set(excluded.final_label.unique()) - {"Excluded"}
    if bad:
        raise ValueError(f"expected only Excluded studies, found {sorted(bad)}")
    out = excluded.copy()
    out["final_label"] = "Other"
    return out


def tag(df: pd.DataFrame, role: str, source: str) -> pd.DataFrame:
    out = df.drop(columns=["split"], errors="ignore").copy()
    out["role"], out["source"] = role, source
    return out


def make_folds(table: pd.DataFrame) -> np.ndarray:
    """Version 1's splitter, now stratified on four classes. A patient's studies are always in the same fold."""
    y = table["final_label"].map({c: i for i, c in enumerate(CLASSES)})
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(table), -1)
    for i, (_, held) in enumerate(skf.split(table, y, groups=table["subject_id"])):
        fold[held] = i
    assert (fold >= 0).all()
    return fold


def check_disjoint(train: pd.DataFrame, val: pd.DataFrame, heldout: pd.DataFrame) -> None:
    """No patient in two roles, and no study twice. Raises rather than quietly continuing."""
    roles = {"train": set(train.subject_id), "val": set(val.subject_id), "heldout": set(heldout.subject_id)}
    for a, b in (("train", "val"), ("train", "heldout"), ("val", "heldout")):
        shared = roles[a] & roles[b]
        if shared:
            raise ValueError(f"{len(shared)} patients are in both {a} and {b}")
    studies = pd.concat([train.study_id, val.study_id, heldout.study_id])
    if studies.duplicated().any():
        raise ValueError(f"{int(studies.duplicated().sum())} studies appear more than once")


def build(train: pd.DataFrame, ex_train: pd.DataFrame, val: pd.DataFrame, ex_val: pd.DataFrame, ex_none: pd.DataFrame) -> dict:
    """The three four-class tables. Pure: nothing is read or written here, so the rules can be tested without the data."""
    t = pd.concat([tag(train, "train", "in_scope"), tag(label_other(ex_train), "train", "excluded")], ignore_index=True)
    v = pd.concat([tag(val, "val", "in_scope"), tag(label_other(ex_val), "val", "excluded")], ignore_index=True)
    h = tag(label_other(ex_none), "heldout", "excluded")
    check_disjoint(t, v, h)
    t["fold"] = make_folds(t)
    return {"train": t, "val": v, "heldout": h}


def counts(tables: dict) -> dict:
    out = {}
    for name, df in tables.items():
        out[name] = {"studies": int(len(df)), "patients": int(df.subject_id.nunique()),
                     "by_class": {c: int((df.final_label == c).sum()) for c in CLASSES}}
    t = tables["train"]
    out["train_folds"] = {str(f): {c: int(((t.fold == f) & (t.final_label == c)).sum()) for c in CLASSES} for f in range(N_FOLDS)}
    return out


def agreement_with_version_1(train_table: pd.DataFrame, v1_train: pd.DataFrame) -> dict:
    """How many in-scope training rows kept the fold version 1 gave them. The folds are not identical, so this is reported, not hidden."""
    sys.path.insert(0, str(ROOT / "scripts"))
    v1 = v1_train.copy()
    v1["y"] = v1["final_label"].map({"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2})
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    old = np.full(len(v1), -1)
    for i, (_, held) in enumerate(skf.split(v1, v1["y"], groups=v1["subject_id"])):
        old[held] = i
    old_by_study = dict(zip(v1.study_id, old))
    ins = train_table[train_table.source == "in_scope"]
    same = np.mean([old_by_study[s] == f for s, f in zip(ins.study_id, ins.fold)])
    return {"in_scope_rows": int(len(ins)), "share_with_the_same_fold": round(float(same), 3),
            "note": "fold numbers are arbitrary labels, so a low share does not by itself mean a different partition of patients"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    read = lambda n: pd.read_csv(PROC / f"{n}.csv")                                # noqa: E731
    train, val = read("train"), read("val")
    tables = build(train, read("excluded_train"), val, read("excluded_val"), read("excluded_none"))
    for name, df in tables.items():
        df.to_csv(PROC / f"v2_{name}.csv", index=False)
    result = counts(tables)
    result["fold_agreement_with_version_1"] = agreement_with_version_1(tables["train"], train)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
