"""
Runs a negation-aware keyword labeler over the radiology report for every linked
study, producing per-study no_finding / pneumonia_mentioned / cancer_mentioned flags.
"""
import csv
import spacy
from negspacy.negation import Negex  # noqa: F401 -- import registers the "negex" pipe factory

RESOLVED_LINKS = "data/interim/resolved_study_links.csv"
REPORTS_ROOT = "data/raw/mimic-cxr/mimic-cxr-dataset/mimic-cxr-reports/files"
OUTPUT = "data/interim/report_labels.csv"

PNEUMONIA_TERMS = ["pneumonia", "consolidation", "infiltrate", "pneumonic"]
CANCER_TERMS = ["mass", "nodule", "malignancy", "neoplasm", "carcinoma", "tumor", "metastasis", "metastases", "malignant"]
OTHER_ABNORMAL_TERMS = ["effusion", "pneumothorax", "edema", "cardiomegaly", "atelectasis",
                        "opacity", "opacities", "fracture", "lesion", "nodular"]
ALL_TERMS = PNEUMONIA_TERMS + CANCER_TERMS + OTHER_ABNORMAL_TERMS

nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "attribute_ruler"])
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([{"label": "FINDING", "pattern": term} for term in ALL_TERMS])
nlp.add_pipe("negex", config={"ent_types": ["FINDING"]})


def report_path(subject_id, study_id):
    sid = str(subject_id)
    return f"{REPORTS_ROOT}/p{sid[:2]}/p{sid}/s{study_id}.txt"


def load_rows():
    with open(RESOLVED_LINKS, newline="") as f:
        return list(csv.DictReader(f))


def label_doc(doc):
    positive_terms = set()
    for ent in doc.ents:
        if ent.label_ == "FINDING" and not ent._.negex:
            positive_terms.add(ent.text.lower())
    return {
        "pneumonia_mentioned": any(t in positive_terms for t in PNEUMONIA_TERMS),
        "cancer_mentioned": any(t in positive_terms for t in CANCER_TERMS),
        "no_finding": len(positive_terms) == 0,
    }


rows = load_rows()
print(f"Labeling reports for {len(rows)} studies...")

texts = []
missing = 0
for r in rows:
    path = report_path(r["subject_id"], r["study_id"])
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    except FileNotFoundError:
        texts.append("")
        missing += 1

print(f"Missing report files: {missing} / {len(rows)}")

results = []
for r, doc in zip(rows, nlp.pipe(texts, batch_size=200)):
    labels = label_doc(doc)
    results.append({
        "subject_id": r["subject_id"],
        "study_id": r["study_id"],
        "hadm_id": r["hadm_id"],
        **labels,
    })

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["subject_id", "study_id", "hadm_id", "pneumonia_mentioned", "cancer_mentioned", "no_finding"])
    writer.writeheader()
    writer.writerows(results)

n_no_finding = sum(r["no_finding"] for r in results)
n_pneumonia = sum(r["pneumonia_mentioned"] for r in results)
n_cancer = sum(r["cancer_mentioned"] for r in results)
print(f"\nNo Finding: {n_no_finding}")
print(f"Pneumonia mentioned: {n_pneumonia}")
print(f"Cancer mentioned: {n_cancer}")
print(f"\nSaved to {OUTPUT}")
