"""
Validates the faithfulness checker itself, because a checker nobody tested is just another
unverified component. A hand-made set of sentences, each labelled faithful or unfaithful to ONE
passage, is scored by faithfulness.check_sentence, and we report how many unfaithful sentences
it catches and how many faithful ones it wrongly drops.

The passages are real retrieved passages (saved from two real validation cases). Sentences are
of three kinds: real outputs of the writer (labelled by reading them against the passage),
paraphrases written to be faithful, and sentences written to be unfaithful in a specific way
(changed number, reversed direction, wrong claim for the passage, overreach, advice, an
unsupported fact). Honest limits: about 30 sentences, written by the project author, the
unfaithful ones are the kinds of error the author thought of. Treat the rates as a sanity
check on the checker, not as a measured error rate for real writer output.

Run:  python -m app.rag.eval_faithfulness   (needs experiments/rag_retrieval/faithfulness_passages.json,
      written by the first run from the saved real cases; see build_passages, which takes the saved
      cases file and the study ids of the pneumonia and the cancer case: main(src, pneumonia_study, cancer_study))
"""
from __future__ import annotations

import torch  # noqa: F401

import json
import sys
from pathlib import Path

from app.rag import faithfulness

OUT = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval"
PASSAGES = OUT / "faithfulness_passages.json"

# (passage key, sentence, faithful?, kind)
SET = [
    ("lymph", "Patients with pneumonia who have a lymphocyte count of 1-2 x 10^9 cells/L are at a significantly increased risk of mortality.", True, "real writer output"),
    ("lymph", "Low lymphocyte counts are associated with higher short- and long-term mortality in pneumonia.", True, "paraphrase"),
    ("lymph", "Even lymphocyte counts within the normal range were associated with increased mortality in a primary care pneumonia cohort.", True, "paraphrase"),
    ("lymph", "High lymphocyte counts are associated with increased mortality in pneumonia.", False, "reversed direction"),
    ("lymph", "Lymphocyte counts of 5-8 x 10^9 cells/L carried significantly increased mortality in pneumonia.", False, "changed number"),
    ("lymph", "Low lymphocyte count causes death in pneumonia.", False, "overreach"),
    ("lymph", "Patients with a low lymphocyte count should be given antibiotics early.", False, "advice"),
    ("lymph", "Elevated C-reactive protein is the strongest predictor of mortality in pneumonia.", False, "unsupported claim"),
    ("albumin", "Low albumin levels have been associated with adverse clinical outcomes in patients with pneumonia.", True, "real writer output"),
    ("albumin", "Serum albumin falls during systemic inflammation and infection.", True, "paraphrase"),
    ("albumin", "Low albumin has been linked to prolonged hospital stays, organ failure and increased mortality.", True, "paraphrase"),
    ("albumin", "Elevated fibrinogen has been associated with poor outcomes in stroke-associated pneumonia and some malignancies.", True, "paraphrase"),
    ("albumin", "High albumin levels have been linked to increased mortality.", False, "reversed direction"),
    ("albumin", "Low albumin was linked to organ failure in 95% of patients.", False, "changed number"),
    ("albumin", "Albumin is a better predictor of mortality than fibrinogen.", False, "unsupported claim"),
    ("albumin", "CURB-65 uses albumin as one of its five factors.", False, "wrong claim for passage"),
    ("albumin", "Doctors should measure albumin and treat low levels with supplements.", False, "advice"),
    ("predictors", "Lymphocyte count, serum albumin and procalcitonin were significant predictors of in-hospital mortality in community-acquired pneumonia.", True, "real writer output"),
    ("predictors", "Lymphocyte count and albumin had the highest predictive accuracy among the markers studied.", True, "paraphrase"),
    ("predictors", "The neutrophil-lymphocyte ratio was a significant predictor of prolonged hospital stay in community-acquired pneumonia.", True, "paraphrase"),
    ("predictors", "Procalcitonin had the highest predictive accuracy of all markers.", False, "wrong claim for passage"),
    ("predictors", "Albumin predicted mortality in 80% of cases of community-acquired pneumonia.", False, "changed number"),
    ("predictors", "These markers prove that every patient with low albumin will die.", False, "overreach"),
    ("nlr_intro", "Elevated neutrophil-to-lymphocyte ratio (NLR) is associated with increased mortality risk in community-acquired pneumonia, with in-hospital mortality rates ranging from 5% to 17% in general populations and up to 32-49% in patients with severe CAP.", False, "real writer output: figures belong to pneumonia, not to NLR"),
    ("nlr_intro", "In community-acquired pneumonia, in-hospital mortality ranges from 5% to 17% in general populations and up to 32-49% in severe cases.", True, "paraphrase"),
    ("nlr_intro", "The neutrophil-to-lymphocyte ratio has emerged as a promising prognostic biomarker in infectious diseases.", True, "paraphrase"),
    ("nlr_intro", "Neutrophil counts typically rise and lymphocyte counts fall under pathological stress, raising the NLR.", True, "paraphrase"),
    ("nlr_intro", "C-reactive protein and procalcitonin are always reliable and inexpensive markers.", False, "reversed claim"),
    ("nlr_intro", "A high NLR means the patient has lung cancer.", False, "unsupported claim"),
    ("adeno", "Invasive mucinous adenocarcinoma can present on chest X-ray as an air-space filling pattern with focal or lobar consolidation.", True, "real writer output"),
    ("adeno", "Air bronchograms are commonly seen in the air-space filling pattern of mucinous adenocarcinoma.", True, "paraphrase"),
    ("adeno", "A central mass or irregular bronchial narrowing can help distinguish lung cancer from pneumonia.", True, "paraphrase"),
    ("adeno", "Lung cancer lacking a response to antibiotics should be treated with chemotherapy.", False, "advice"),
    ("adeno", "Mucinous adenocarcinoma never causes consolidation on chest X-ray.", False, "reversed direction"),
    ("adeno", "Smoking is the main cause of invasive mucinous adenocarcinoma.", False, "unsupported claim"),
]

# Which saved case and passage number each key came from. The cases are named by role, not by study id, because study ids come from MIMIC
# and stay out of the repository; build_passages is told the two ids on the command line.
KEYS = {"lymph": ("pneumonia", 2), "albumin": ("pneumonia", 3), "predictors": ("pneumonia", 4),
        "nlr_intro": ("pneumonia", 1), "adeno": ("cancer", 1)}


def build_passages(src: Path, cases: dict) -> dict:
    data = json.loads(src.read_text(encoding="utf-8"))
    out = {k: next(p["text"] for p in data[cases[role]]["passages"] if p["n"] == n) for k, (role, n) in KEYS.items()}
    PASSAGES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(src: str | None = None, pneumonia_study: str | None = None, cancer_study: str | None = None) -> None:
    passages = (json.loads(PASSAGES.read_text(encoding="utf-8")) if PASSAGES.exists()
                else build_passages(Path(src), {"pneumonia": pneumonia_study, "cancer": cancer_study}))
    rows = []
    for key, sent, ok, kind in SET:
        ent = faithfulness.entailment(sent, passages[key])
        nums = sorted(faithfulness._numbers(sent) - faithfulness._numbers(passages[key]))
        banned = sorted({m.group(0).lower() for m in faithfulness.BANNED.finditer(sent)})
        rows.append({"key": key, "sentence": sent, "faithful": ok, "kind": kind, "entailment": round(ent, 3),
                     "bad_numbers": nums, "banned": banned})
    print(f"{'faithful':<9}{'ent':>6}  {'num':<4}{'ban':<4} kind / sentence")
    for r in rows:
        print(f"{str(r['faithful']):<9}{r['entailment']:>6.2f}  {'X' if r['bad_numbers'] else '-':<4}"
              f"{'X' if r['banned'] else '-':<4} {r['kind']}: {r['sentence'][:70]}")
    print()
    n_f = sum(r["faithful"] for r in rows)
    n_u = len(rows) - n_f
    for thr in (0.3, 0.5, 0.7, 0.9, 0.95):
        def passes(r):
            return r["entailment"] >= thr and not r["bad_numbers"] and not r["banned"]
        caught = sum(1 for r in rows if not r["faithful"] and not passes(r))
        kept = sum(1 for r in rows if r["faithful"] and passes(r))
        print(f"threshold {thr}: catches {caught}/{n_u} unfaithful, keeps {kept}/{n_f} faithful")
    (OUT / "faithfulness_eval.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(*sys.argv[1:])
