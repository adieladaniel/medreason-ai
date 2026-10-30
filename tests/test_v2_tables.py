"""Version 2, step 1 (scripts/v2_build_tables.py): the roles, the fourth class and the folds. The rules are tested on synthetic tables; one
check on the real tables skips when the git-ignored data is absent."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("v2_build_tables", ROOT / "scripts" / "v2_build_tables.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def table(first_subject, n_patients, labels, studies_per_patient=3, split=None, start_study=0):
    rows, sid = [], start_study
    for p in range(n_patients):
        for k in range(studies_per_patient):
            rows.append({"subject_id": first_subject + p, "study_id": sid, "hadm_id": sid, "final_label": labels[(p + k) % len(labels)],
                         "image_path": f"i{sid}.png", "report_path": f"r{sid}.txt", "wbc": 1.0, **({"split": split} if split else {})})
            sid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def parts():
    in_labels = ["Normal", "Pneumonia", "Pneumonia", "Lung Cancer"]
    train = table(0, 80, in_labels, split="train")
    val = table(1000, 20, in_labels, split="val", start_study=10_000)
    # Excluded studies of training and validation patients belong to patients who also have in-scope studies; the held-out ones do not
    ex_train = table(0, 30, ["Excluded"], studies_per_patient=2, start_study=20_000)
    ex_val = table(1000, 8, ["Excluded"], studies_per_patient=2, start_study=30_000)
    ex_none = table(5000, 25, ["Excluded"], studies_per_patient=2, start_study=40_000)
    return train, ex_train, val, ex_val, ex_none


def test_excluded_studies_become_the_fourth_class_and_nothing_else_is_relabelled(parts):
    t = v2.build(*parts)
    assert set(t["train"].final_label) == set(v2.CLASSES)
    assert (t["train"][t["train"].source == "excluded"].final_label == "Other").all()
    assert not (t["train"][t["train"].source == "in_scope"].final_label == "Other").any()
    assert (t["heldout"].final_label == "Other").all() and (t["heldout"].role == "heldout").all()
    assert set(t["val"].role) == {"val"}


def test_the_counts_are_the_sum_of_the_parts(parts):
    train, ex_train, val, ex_val, ex_none = parts
    t = v2.build(*parts)
    assert len(t["train"]) == len(train) + len(ex_train)
    assert len(t["val"]) == len(val) + len(ex_val)
    assert len(t["heldout"]) == len(ex_none)


def test_a_patient_is_in_one_fold_only_and_every_study_has_a_fold(parts):
    t = v2.build(*parts)["train"]
    assert (t.fold >= 0).all() and (t.fold < v2.N_FOLDS).all()
    assert (t.groupby("subject_id").fold.nunique() == 1).all()
    assert set(t.fold) == set(range(v2.N_FOLDS))


def test_every_fold_holds_every_class(parts):
    t = v2.build(*parts)["train"]
    for f in range(v2.N_FOLDS):
        assert set(t[t.fold == f].final_label) == set(v2.CLASSES), f


def test_the_folds_are_reproducible_and_the_inputs_are_not_changed(parts):
    before = [p.copy() for p in parts]
    a, b = v2.build(*parts)["train"], v2.build(*parts)["train"]
    assert np.array_equal(a.fold.values, b.fold.values)
    for p, q in zip(parts, before):
        pd.testing.assert_frame_equal(p, q)


def test_the_split_column_of_version_1_is_dropped(parts):
    t = v2.build(*parts)
    assert all("split" not in df.columns for df in t.values())


@pytest.mark.parametrize("which, other", [("heldout_with_train_patient", 0), ("heldout_with_val_patient", 1000)])
def test_a_patient_in_two_roles_stops_the_build(parts, which, other):
    train, ex_train, val, ex_val, ex_none = parts
    ex_none = ex_none.copy()
    ex_none.loc[0, "subject_id"] = other                      # a held-out study of a patient who is in another role
    with pytest.raises(ValueError, match="patients are in both"):
        v2.build(train, ex_train, val, ex_val, ex_none)


def test_a_study_used_twice_stops_the_build(parts):
    train, ex_train, val, ex_val, ex_none = parts
    ex_none = ex_none.copy()
    ex_none.loc[0, "study_id"] = int(train.study_id.iloc[0])
    with pytest.raises(ValueError, match="more than once"):
        v2.build(train, ex_train, val, ex_val, ex_none)


def test_a_table_that_is_not_all_excluded_stops_the_build(parts):
    train, ex_train, val, ex_val, ex_none = parts
    bad = ex_train.copy()
    bad.loc[0, "final_label"] = "Pneumonia"
    with pytest.raises(ValueError, match="only Excluded"):
        v2.build(train, bad, val, ex_val, ex_none)


def test_this_script_has_no_way_to_name_the_test_table():
    code = (ROOT / "scripts" / "v2_build_tables.py").read_text(encoding="utf-8")
    assert "test.csv" not in code and 'read("test")' not in code and "excluded_test" not in code


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "v2_train.csv").exists(), reason="needs the built git-ignored tables")
def test_the_real_tables_match_the_plan_and_hold_no_test_patient():
    proc = ROOT / "data" / "processed"
    t, v, h = (pd.read_csv(proc / f"v2_{n}.csv") for n in ("train", "val", "heldout"))
    assert (len(t), len(v), len(h)) == (11039 + 5369, 2315 + 1234, 2241)
    assert t.final_label.value_counts().to_dict() == {"Pneumonia": 7312, "Other": 5369, "Normal": 2953, "Lung Cancer": 774}
    test_patients = set(pd.read_csv(proc / "test.csv", usecols=["subject_id"]).subject_id)
    for df in (t, v, h):
        assert not set(df.subject_id) & test_patients
    assert (t.groupby("subject_id").fold.nunique() == 1).all()
