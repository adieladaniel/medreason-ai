"""
The arithmetic of scripts/eval_other_pathology.py (rates, the confidence-as-detector AUROC, the in-scope false-alarm rule, subgroups)
on hand-made records, and the promises about data: the Excluded set is built only from patients the system has not trained on and
who are not in the sealed split, and the default dataset build is unchanged.
"""
from pathlib import Path

import pandas as pd
import pytest

from scripts import eval_other_pathology as op

ROOT = Path(__file__).resolve().parent.parent


def rec(group="excluded", cls="Pneumonia", p=0.9, low=False, size=1, alert=None, true=None, subject=1, mods="blood+image+text"):
    return {"group": group, "subject_id": subject, "true_label": true, "predicted_class": cls, "max_prob": p, "p_cancer": 0.1,
            "modalities": mods, "low_confidence": low, "set_size": size, "alert": alert}


def test_a_prediction_becomes_a_flat_record():
    pred = {"belief": {"predicted_class": "Normal", "probabilities": {"Normal": 0.7, "Pneumonia": 0.2, "Lung Cancer": 0.1}},
            "modalities_used": ["text", "image"],
            "decision": {"low_confidence": True, "prediction_set": ["Normal", "Pneumonia"], "cancer_alert": {"informative": True, "flag": True}}}
    r = op.record_from_prediction("excluded", 7, pred)
    assert r["predicted_class"] == "Normal" and r["max_prob"] == 0.7 and r["modalities"] == "image+text"
    assert r["low_confidence"] is True and r["set_size"] == 2 and r["alert"] is True
    pred["decision"]["cancer_alert"] = {"informative": False, "flag": True}          # an alert that is not offered counts as none
    assert op.record_from_prediction("excluded", 7, pred)["alert"] is None


def test_report_mentions_are_plain_word_matches():
    m = op.mentions("Small left pleural effusion. Mild pulmonary edema. No pneumothorax. Atelectatic change.")
    assert m == {"effusion": True, "edema": True, "atelectasis": True, "pneumothorax": True, "cardiomegaly": False, "fracture": False}
    assert not any(op.mentions("The lungs are clear.").values())


def test_rates_count_what_they_say():
    recs = [rec(cls="Pneumonia", p=0.9, low=False, size=1), rec(cls="Pneumonia", p=0.6, low=True, size=2),
            rec(cls="Normal", p=0.86, low=False, size=1, alert=False), rec(cls="Lung Cancer", p=0.5, low=True, size=3, alert=True)]
    r = op._rates(recs)
    assert r["names_pneumonia"] == 0.5 and r["names_normal"] == 0.25 and r["names_lung_cancer"] == 0.25
    assert r["not_low_confidence"] == 0.5 and r["at_least_85"] == 0.5 and r["single_class_set"] == 0.5
    assert r["alert"] == 0.5                                                          # 1 of the 2 studies where an alert was offered
    assert op._rates([]) == {}


def test_the_in_scope_alert_rate_leaves_out_true_cancers():
    inside = [rec("in_scope", true="Normal", alert=True), rec("in_scope", true="Normal", alert=False),
              rec("in_scope", true="Lung Cancer", alert=True), rec("in_scope", true="Lung Cancer", alert=True)]
    s = op._stats(inside, [rec(alert=True)])
    assert s["in_alert"] == 0.5 and s["out_alert"] == 1.0                              # not 0.75: the true cancers are not false alarms


def test_confidence_detects_out_of_scope_exactly_when_it_is_lower_there():
    inside = [rec("in_scope", p=0.9, subject=i) for i in range(10)]
    low = [rec(p=0.5, subject=100 + i) for i in range(10)]
    same = [rec(p=0.9, subject=100 + i) for i in range(10)]
    assert op._stats(inside, low)["auroc_confidence_detects_out_of_scope"] == 1.0
    assert op._stats(inside, same)["auroc_confidence_detects_out_of_scope"] == 0.5
    assert op._stats(inside, [rec(p=0.95, subject=100 + i) for i in range(10)])["auroc_confidence_detects_out_of_scope"] == 0.0


def test_the_summary_has_intervals_that_contain_the_estimate_and_resample_patients():
    inside = [rec("in_scope", true="Normal", p=0.9 - 0.01 * (i % 7), low=i % 5 == 0, subject=i) for i in range(60)]
    outside = [rec(p=0.8 - 0.01 * (i % 9), low=i % 3 == 0, subject=1000 + i // 2) for i in range(60)]
    s = op.summarise(inside + outside, b=200)
    assert s["n_in_scope"] == 60 and s["n_excluded"] == 60 and s["patients_excluded"] == 30
    for k, (lo, hi) in s["interval"].items():
        assert lo <= s["point"][k] <= hi or abs(s["point"][k] - lo) < 1e-9 or abs(s["point"][k] - hi) < 1e-9, k
    assert s == op.summarise(inside + outside, b=200)                                   # same seed, same answer
    assert s["excluded_modalities"] == {"blood+image+text": 60}


def test_subgroups_report_only_kinds_that_occur_and_check_alignment():
    recs = [rec(cls="Pneumonia", p=0.9), rec(cls="Normal", p=0.7, low=True), rec(cls="Pneumonia", p=0.95)]
    flags = pd.DataFrame({"effusion": [True, False, True], "fracture": [False, False, False], "tuberculosis_coded": [False, True, False]})
    g = op.subgroup(recs, flags)
    assert set(g) == {"effusion", "tuberculosis_coded"} and g["effusion"]["n"] == 2 and g["effusion"]["names_pneumonia"] == 1.0
    assert g["tuberculosis_coded"]["not_low_confidence"] == 0.0
    with pytest.raises(AssertionError):
        op.subgroup(recs, flags.iloc[:2])


def test_the_evaluation_never_names_the_sealed_split():
    assert "test.csv" not in (ROOT / "scripts" / "eval_other_pathology.py").read_text(encoding="utf-8")


def test_the_excluded_build_keeps_only_patients_in_neither_training_nor_the_sealed_split_and_the_default_is_unchanged():
    src = (ROOT / "scripts" / "build_final_dataset.py").read_text(encoding="utf-8")
    assert '"f.final_label != \'Excluded\'"' in src or "f.final_label != 'Excluded'" in src       # the default build
    assert "split_subjects(\"train\")" in src and "split_subjects(\"test\")" in src and "everyone - (" in src     # unseen = all patients minus train and test
    assert "f.final_label = 'Excluded' AND f.subject_id IN (SELECT subject_id FROM chosen)" in src
    assert 'default=OUTPUT' in src
