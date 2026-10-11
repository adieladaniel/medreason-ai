"""
Evaluation queries for the Phase 6 retriever (stage 2, 27-paper corpus).

A passage is relevant to a query if (1) its paper's topic tag is in the query's allowed
topics and (2) its text matches every regex group in `must`. This replaces the stage-1
fixed list of acceptable papers, which would have penalised new papers that genuinely
answer a question. It is still a keyword proxy: containing the terms is necessary for
relevance, not sufficient, and app/rag/eval_pipeline.py reports an audited estimate of how
often the proxy agrees with a human reading.

Queries come in two styles: a clinician's question, and the terse keyword form the evidence
planner will actually emit from a prediction plus its top attributions. `grp` ties together
queries about the same fact so they always fall in the same split (dev or test), otherwise a
question and its keyword twin would leak across the split.

Written by the project author with the assistant after reading the papers' titles and
abstracts, so queries share vocabulary with the sources to a degree an independent
annotator's would not. Treat absolute numbers as optimistic; differences between pipelines
on the same set are more trustworthy than the levels.
"""
from __future__ import annotations

PNEU = {"pneumonia", "differential"}
CANCER = {"lung_cancer", "lung_cancer_labs", "differential"}
BOTH = PNEU | CANCER


def q(grp, kind, text, topics, *must):
    return {"grp": grp, "kind": kind, "q": text, "topics": topics, "must": list(must)}


QUERIES = [
    # ---- pneumonia: imaging
    q("pneu_cxr", "question", "What are the radiographic features of community-acquired pneumonia?", PNEU, r"consolidat|infiltrat|opacit"),
    q("pneu_cxr", "case", "pneumonia, lung opacity and consolidation on chest X-ray", PNEU, r"consolidat|opacit"),
    q("pneu_cxr_ct", "question", "How reliable is the chest radiograph for diagnosing pneumonia compared with CT?", PNEU, r"radiograph|chest x-ray|CXR", r"\bCT\b|computed tomography"),
    q("pneu_icu_cxr", "case", "chest radiograph findings in pneumonia patients needing ICU admission", PNEU, r"radiograph|chest x-ray|CXR", r"ICU|intensive"),
    # ---- pneumonia: inflammatory markers
    q("crp", "question", "Is C-reactive protein useful to diagnose community-acquired pneumonia?", PNEU, r"C-reactive|CRP", r"sensitiv|specific|accura|AUC|area under"),
    q("crp", "case", "pneumonia with leukocytosis and elevated CRP", PNEU, r"leukocyt|white blood|WBC|C-reactive|CRP"),
    q("pct", "question", "Can procalcitonin help identify bacterial pneumonia?", PNEU, r"procalcitonin|PCT", r"bacteri"),
    q("pct", "case", "elevated procalcitonin, suspected bacterial pneumonia", PNEU, r"procalcitonin|PCT"),
    q("wbc", "case", "upper lobe consolidation with elevated white blood cell count", PNEU, r"consolidat|pneumonia", r"leukocyt|white blood|WBC"),
    q("wbc", "question", "Does leukocytosis help diagnose pneumonia?", PNEU, r"leukocyt|white blood|WBC", r"pneumonia"),
    # ---- pneumonia: blood-count ratios and chemistry
    q("nlr_pneu", "question", "Does a high neutrophil-to-lymphocyte ratio predict mortality in pneumonia?", PNEU, r"neutrophil", r"mortality|death"),
    q("nlr_pneu", "case", "pneumonia with elevated neutrophil-lymphocyte ratio", PNEU, r"NLR|neutrophil.{0,25}lymphocyte"),
    q("lymph_pneu", "question", "How does lymphopenia affect outcomes in pneumonia?", PNEU, r"lymphopeni|lymphocyt", r"mortality|outcome|severity"),
    q("lymph_pneu", "case", "pneumonia with low lymphocyte count", PNEU, r"lymphopeni|lymphocyt"),
    q("urea", "question", "Is blood urea a predictor of pneumonia severity?", PNEU, r"urea|\bBUN\b", r"severity|mortality|CURB"),
    q("urea", "case", "pneumonia with elevated urea, CURB-65 severity", PNEU, r"CURB|urea"),
    q("curb", "question", "How well do CURB-65 and the pneumonia severity index predict mortality?", PNEU, r"CURB-65", r"mortality"),
    q("alb_pneu", "question", "Does low serum albumin predict worse pneumonia outcomes?", PNEU, r"albumin", r"mortality|outcome|severity"),
    q("alb_pneu", "case", "pneumonia with low albumin", PNEU, r"albumin"),
    q("plr_pneu", "question", "Is the platelet-to-lymphocyte ratio useful in pneumonia?", PNEU, r"platelet.{0,25}lymphocyte|PLR", r"pneumonia"),
    q("routine_pneu", "question", "Which routine blood tests predict outcomes in community-acquired pneumonia?", PNEU, r"albumin|lymphocyt|urea|hemoglobin|platelet", r"mortality|outcome"),
    # ---- pneumonia versus cancer, and other differentials
    q("persist", "question", "Does a consolidation that persists after pneumonia treatment suggest lung cancer?", BOTH, r"consolidat|persist", r"cancer|malignan|carcinoma"),
    q("persist", "case", "consolidation not resolving after antibiotics, possible malignancy", BOTH, r"consolidat|resol|persist", r"malignan|cancer|carcinoma"),
    q("postobs", "question", "How is post-obstructive pneumonia different from community-acquired pneumonia?", {"differential", "pneumonia"}, r"obstructi"),
    q("postobs", "case", "lobar atelectasis with consolidation, central tumor obstructing a bronchus", {"differential", "lung_cancer"}, r"obstruct|atelectasis", r"tumou?r|mass|cancer|endobronchial"),
    q("edema", "question", "How can pulmonary edema be told apart from pneumonia on a chest film?", PNEU, r"edema", r"cardiomegaly|effusion|pneumonia"),
    q("edema", "case", "pulmonary edema with cardiomegaly and pleural effusion", PNEU, r"edema", r"cardiomegaly|effusion"),
    q("effusion", "question", "How are parapneumonic effusions and empyema recognised on imaging?", {"differential", "pneumonia"}, r"effusion|empyema", r"pneumonia|parapneumonic|pleural"),
    q("effusion", "case", "pleural effusion with pneumonia", {"differential", "pneumonia"}, r"effusion"),
    q("mimic", "question", "Can lung adenocarcinoma look like pneumonia?", BOTH, r"adenocarcinoma", r"pneumonia"),
    q("mimic", "case", "adenocarcinoma presenting as pneumonia", BOTH, r"adenocarcinoma", r"pneumonia"),
    q("c_vs_i", "question", "Which imaging findings suggest cancer rather than infection in a lung consolidation?", BOTH, r"consolidat|opacit", r"cancer|malignan|adenocarcinoma"),
    # ---- lung cancer: imaging
    q("ca_cxr", "question", "What does lung cancer look like on a chest X-ray?", CANCER, r"mass|nodule", r"radiograph|chest x-ray|CXR|opacit"),
    q("ca_cxr", "case", "lung mass on chest radiograph with hilar enlargement", CANCER, r"hilar|\bmass"),
    q("missed", "question", "Why are lung cancers missed on chest radiographs?", CANCER, r"miss|overlook|blind.?spot"),
    q("missed", "case", "lung lesion at the apex or behind the heart overlooked on a chest film", CANCER, r"apex|apical|retrocardiac|behind the heart|overlap"),
    q("nodule", "question", "How are pulmonary nodules larger than 3 cm classified and managed?", CANCER, r"3 ?cm|30 ?mm|\bmass"),
    q("nodule", "case", "spiculated pulmonary nodule, lung cancer risk", CANCER, r"spicul|nodule"),
    # ---- lung cancer: laboratory markers
    q("ldh", "question", "Is elevated LDH linked to worse survival in lung cancer?", CANCER, r"LDH|lactate dehydrogenase", r"surviv|prognos|mortality"),
    q("ldh", "case", "lung cancer with elevated LDH", CANCER, r"LDH|lactate dehydrogenase"),
    q("ldh_alb", "case", "lung cancer with elevated LDH and low albumin", CANCER, r"LDH|lactate dehydrogenase", r"albumin"),
    q("alb_ca", "question", "Does low serum albumin predict outcome in lung cancer?", CANCER, r"albumin", r"surviv|prognos"),
    q("nlr_ca", "question", "What is the prognostic value of the neutrophil-to-lymphocyte ratio in non-small cell lung cancer?", CANCER, r"NLR|neutrophil.{0,25}lymphocyte", r"surviv|prognos"),
    q("nlr_ca", "case", "high neutrophil-lymphocyte ratio and platelet-lymphocyte ratio in lung cancer", CANCER, r"NLR|PLR|lymphocyte ratio"),
    q("plt_ca", "question", "Does a high platelet count predict poor survival in lung cancer?", CANCER, r"platelet|thrombocyt", r"surviv|prognos"),
    q("plt_ca", "case", "lung cancer with thrombocytosis", CANCER, r"thrombocytosis|platelet count"),
    q("anemia", "question", "How common is anemia in lung cancer and does it affect survival?", CANCER, r"anemi|anaemi|hemoglobin", r"surviv|prognos|prevalen"),
    q("anemia", "case", "lung cancer with low hemoglobin", CANCER, r"anemi|anaemi|hemoglobin"),
    q("halp", "question", "What is the HALP score and what does it predict in lung cancer?", CANCER, r"HALP"),
    q("sodium", "question", "Why does small cell lung cancer cause a low sodium level?", CANCER, r"hyponatremi|SIADH|sodium"),
    q("sodium", "case", "small cell lung cancer with hyponatremia", CANCER, r"hyponatremi|SIADH"),
    q("inflam_ratios", "question", "Which inflammation-based blood ratios are used to estimate prognosis in lung cancer?", CANCER, r"ratio", r"surviv|prognos"),
    q("malig_eff", "question", "How is a pleural effusion assessed for malignancy on imaging?", {"differential", "lung_cancer"}, r"pleural|effusion", r"malignan"),
    q("malig_eff", "case", "lung cancer with malignant pleural effusion on imaging", {"differential", "lung_cancer"}, r"effusion", r"malignan"),
]

# medical, but outside what the corpus covers: the system should refuse to cite anything.
# Several are deliberately near the corpus (tuberculosis, COPD, PE, pneumothorax) to test the floor hard.
OFF_CORPUS = [
    "management of diabetic ketoacidosis in adults", "recovery after total knee replacement surgery",
    "treatment of hypertension in pregnancy", "inhaler technique in pediatric asthma",
    "MRI findings in multiple sclerosis", "timing of surgery for hip fracture",
    "colorectal cancer screening colonoscopy intervals", "antidepressant dosing in elderly patients",
    "anticoagulant therapy for pulmonary embolism", "chest tube insertion technique for pneumothorax",
    "management of acute COPD exacerbation", "first-line drug regimen for pulmonary tuberculosis",
    "heart failure with preserved ejection fraction treatment", "fluid resuscitation guidelines in septic shock",
    "biologic therapy for severe eosinophilic asthma", "lung transplant rejection immunosuppression",
]

GROUPS = list(dict.fromkeys(x["grp"] for x in QUERIES))
SPLIT = {g: ("dev" if i % 2 == 0 else "test") for i, g in enumerate(GROUPS)}   # alternate by topic group
for _x in QUERIES:
    _x["split"] = SPLIT[_x["grp"]]

if __name__ == "__main__":
    from collections import Counter
    print(len(QUERIES), "queries,", len(GROUPS), "groups,", Counter(x["split"] for x in QUERIES), "|", len(OFF_CORPUS), "off-corpus")
