"""Version 2, step 0 (scripts/v2_step0.py): the safeguards that keep the sealed test split out of this work. The checks on real data skip when
the git-ignored tables are absent, as on a clean checkout."""
import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("v2_step0", ROOT / "scripts" / "v2_step0.py")
step0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step0)


def test_the_roles_this_work_may_ask_for_exclude_the_test_patients():
    assert set(step0.ROLES) == {"train", "val", "none"}


def test_no_version_2_script_or_plan_input_names_the_test_table():
    for path in [ROOT / "scripts" / "v2_step0.py", *sorted((ROOT / "scripts").glob("v2_*.py"))]:
        code = path.read_text(encoding="utf-8")
        assert not re.search(r"processed[/\\\"' ]*test|test\.csv|excluded_test", code), path.name


def test_the_data_files_that_are_hashed_exist_in_the_list_without_duplicates():
    assert len(step0.DATA_FILES) == len(set(step0.DATA_FILES))
    assert not any("test" in Path(f).name for f in step0.DATA_FILES)


def test_sha_is_the_sha256_of_the_whole_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 700_000)                                   # larger than one 1 MiB read
    import hashlib
    assert step0.sha(f) == hashlib.sha256(f.read_bytes()).hexdigest()


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "excluded_none.csv").exists(), reason="needs the git-ignored tables")
def test_no_patient_is_in_two_roles_and_none_of_the_roles_holds_a_test_patient():
    proc = ROOT / "data" / "processed"
    ids = {n: set(pd.read_csv(proc / f"{n}.csv", usecols=["subject_id"]).subject_id)
           for n in ("train", "val", "test", "excluded_train", "excluded_val", "excluded_none")}
    for role in ("excluded_train", "excluded_val", "excluded_none"):
        assert not ids[role] & ids["test"], role
    assert not ids["excluded_val"] & ids["excluded_none"]
    assert not ids["excluded_none"] & (ids["train"] | ids["val"])
    assert ids["excluded_train"] <= ids["train"] and ids["excluded_val"] <= ids["val"]      # these roles hold only patients who also have in-scope studies
    assert not ids["excluded_train"] & ids["val"]
    assert not ids["excluded_val"] & ids["train"]
