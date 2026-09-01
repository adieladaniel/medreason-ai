"""Sanity check the negation-aware labeler against a known report before scaling up."""
import spacy
from negspacy.negation import Negex

PNEUMONIA_TERMS = ["pneumonia", "consolidation", "infiltrate", "pneumonic"]
CANCER_TERMS = ["mass", "nodule", "malignancy", "neoplasm", "carcinoma", "tumor", "metastasis", "metastases", "malignant"]
OTHER_ABNORMAL_TERMS = ["effusion", "pneumothorax", "edema", "cardiomegaly", "atelectasis",
                        "opacity", "opacities", "fracture", "lesion", "nodular"]
ALL_TERMS = PNEUMONIA_TERMS + CANCER_TERMS + OTHER_ABNORMAL_TERMS

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([{"label": "FINDING", "pattern": term} for term in ALL_TERMS])
nlp.add_pipe("negex", config={"ent_types": ["FINDING"]})


def label_report(text):
    doc = nlp(text)
    positive_terms = set()
    negated_terms = set()
    for ent in doc.ents:
        if ent.label_ == "FINDING":
            term = ent.text.lower()
            if ent._.negex:
                negated_terms.add(term)
            else:
                positive_terms.add(term)
    pneumonia = any(t in positive_terms for t in PNEUMONIA_TERMS)
    cancer = any(t in positive_terms for t in CANCER_TERMS)
    no_finding = len(positive_terms) == 0
    return {
        "pneumonia_positive": pneumonia,
        "cancer_positive": cancer,
        "no_finding": no_finding,
        "positive_terms": positive_terms,
        "negated_terms": negated_terms,
    }


test_report = """
FINAL REPORT
EXAM: Chest frontal and lateral views.
CLINICAL INFORMATION: Wheeze.
COMPARISON: None.
FINDINGS: Frontal and lateral views of the chest were obtained. Lungs are
clear without focal consolidation. No pleural effusion or pneumothorax is
seen. Cardiac and mediastinal silhouettes are unremarkable. No overt
pulmonary edema is seen.
IMPRESSION: No acute cardiopulmonary process.
"""

print(label_report(test_report))

# also test a genuinely positive pneumonia report
positive_report = """
FINDINGS: There is a new right lower lobe consolidation concerning for pneumonia.
No pleural effusion.
IMPRESSION: Findings consistent with pneumonia.
"""
print(label_report(positive_report))
