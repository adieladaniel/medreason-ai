"""
Text unimodal baseline for the 3-class target (Normal / Pneumonia / Lung Cancer).

THE LEAKAGE PROBLEM
-------------------
The dataset's labels are partly derived from the radiology reports themselves
(scripts/exploration/build_final_labels.py: Pneumonia/Cancer fire on a non-negated
keyword mention; Normal *requires* the report to be finding-free). Concretely, of the
label evidence:
  - Pneumonia: 65% of studies have a report-derived mention, 35% are ICD-only
  - Lung Cancer: 54% report-derived, 46% ICD-only
  - Normal: 100% depend on the report being read as clean
So a classifier fed the raw report FINDINGS/IMPRESSION would largely be reading its own
label back. That number is not a capability estimate.

WHAT THIS SCRIPT MEASURES
-------------------------
Two settings, both reported; setting A is the honest one:

  A. indication  -- input = the report's INDICATION / clinical-history section only.
     This is the *referring clinician's* reason for the study (symptoms, pretest
     suspicion), written before the radiologist's read, and never used in label
     derivation. Closest available proxy to "patient presentation" (the text modality's
     original intent). May carry mild residual correlation ("?PNA") -- that is a
     legitimate real-world input, not the diagnostic readout. PRIMARY RESULT.

  B. full_report -- input = the whole report. Reported ONLY as a leakage-inflated
     ceiling / pipeline sanity check. Do not cite it as text-modality performance.

For setting A we also split the val set into report-derived vs ICD-only labels and report
macro F1 on each: similar scores there are evidence A is using real clinical context, not
a residual leak.

ENCODER: frozen Bio_ClinicalBERT (mean-pooled token embeddings), matching the image
baseline's "frozen domain-pretrained backbone + small trainable head" pattern. Head =
multinomial logistic regression (+ a small MLP for comparison), class_weight balanced.
Embeddings are cached to disk; first run downloads ~430 MB of weights and takes a few
minutes on CPU, reruns are instant.

Metric parity: macro F1 on val. The test split is never read.
Run from repo root:  python scripts/train_text_baseline.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path("data/processed")
LABELS_FILE = Path("data/interim/final_study_labels.csv")
OUT_DIR = Path("experiments/text_baseline")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
SEED = 42
MAX_TOKENS = 512
BATCH = 32

LABEL_MAP = {"Normal": 0, "Pneumonia": 1, "Lung Cancer": 2}
LABEL_NAMES = ["Normal", "Pneumonia", "Lung Cancer"]

INDICATION_HEADERS = [
    "INDICATION", "HISTORY", "CLINICAL HISTORY", "CLINICAL INDICATION",
    "CLINICAL INFORMATION", "REASON FOR EXAMINATION", "REASON FOR EXAM",
    "PATIENT HISTORY",
]
_HEADER_RE = r"\n\s*[A-Z][A-Z /()\-]{2,40}:"


def read_report(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def clean(text: str) -> str:
    """Drop the ___ de-identification placeholders and collapse whitespace."""
    text = text.replace("___", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_indication(text: str) -> str:
    """Pull the clinician's reason-for-exam section; '' if the report has none."""
    for hdr in INDICATION_HEADERS:
        m = re.search(rf"{hdr}:\s*(.*?)(?={_HEADER_RE}|\Z)", text, re.S)
        if m and m.group(1).strip():
            frag = m.group(1)
            # INDICATION lines often repeat the text after '//'; keep one copy
            frag = frag.split("//")[0] if "//" in frag else frag
            return clean(frag)
    return ""


@torch.no_grad()
def embed(texts: list[str], tokenizer, model) -> np.ndarray:
    """Mean-pooled (mask-aware) last-hidden-state embedding, frozen, CPU."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = [t if t else "[CLS]" for t in texts[i : i + BATCH]]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        hidden = model(**enc).last_hidden_state            # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy())
        if (i // BATCH) % 20 == 0:
            print(f"    embedded {i + len(batch)}/{len(texts)}")
    return np.vstack(out).astype(np.float32)


def get_embeddings(setting: str, split: str, df: pd.DataFrame, tokenizer, model) -> np.ndarray:
    cache = OUT_DIR / f"emb_{setting}_{split}.npy"
    if cache.exists():
        return np.load(cache)
    print(f"  computing {setting}/{split} embeddings ({len(df)} texts)...")
    raw = [read_report(p) for p in df["report_path"]]
    if setting == "indication":
        texts = [extract_indication(t) for t in raw]
    else:  # full_report
        texts = [clean(t) for t in raw]
    emb = embed(texts, tokenizer, model)
    np.save(cache, emb)
    return emb


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["y"] = df["final_label"].map(LABEL_MAP)
    return df


def eval_model(name: str, clf, X, y_true, results: dict) -> np.ndarray:
    y_pred = clf.predict(X)
    macro = f1_score(y_true, y_pred, average="macro")
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"\n----- {name} (val) -----")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=pred) Normal/Pneumonia/Cancer:\n", cm)
    results[name] = {
        "macro_f1": round(float(macro), 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(LABEL_NAMES, per)},
        "confusion_matrix": cm.tolist(),
    }
    return y_pred


def label_provenance(df: pd.DataFrame, prov: pd.DataFrame) -> pd.Series:
    """Per row: 'report' if the label had report-derived evidence, else 'icd_only'."""
    m = df.merge(prov, on="study_id", how="left", suffixes=("", "_p"))
    is_report = (
        ((m["final_label"] == "Pneumonia") & (m["pneumonia_mentioned"] == True))  # noqa: E712
        | ((m["final_label"] == "Lung Cancer") & (m["cancer_mentioned"] == True))  # noqa: E712
        | (m["final_label"] == "Normal")
    )
    return np.where(is_report, "report", "icd_only")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, val = load_split("train"), load_split("val")
    print(f"train n={len(train)}  val n={len(val)}")

    print(f"loading frozen encoder: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    prov = pd.read_csv(LABELS_FILE)[
        ["study_id", "pneumonia_mentioned", "cancer_mentioned", "no_finding"]
    ]
    val_prov = label_provenance(val, prov)

    results: dict = {
        "encoder": MODEL_NAME,
        "seed": SEED,
        "dataset": {
            "train_n": len(train), "val_n": len(val),
            "val_label_counts": val["final_label"].value_counts().to_dict(),
            "val_provenance_counts": pd.Series(val_prov).value_counts().to_dict(),
        },
        "settings": {},
        "note": (
            "Setting A (indication) is the honest text-modality estimate. Setting B "
            "(full_report) is leakage-inflated: labels were partly derived from the "
            "report text. Cite A, not B."
        ),
    }

    val_pred_export = val[["subject_id", "study_id", "hadm_id", "final_label"]].copy()
    val_pred_export["label_provenance"] = val_prov

    for setting in ["indication", "full_report"]:
        print(f"\n================ setting: {setting} ================")
        Xtr = get_embeddings(setting, "train", train, tokenizer, model)
        Xva = get_embeddings(setting, "val", val, tokenizer, model)
        ytr, yva = train["y"].values, val["y"].values

        setting_res: dict = {}

        logreg = LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED,
        )
        logreg.fit(Xtr, ytr)
        lr_pred = eval_model(f"{setting} / logreg", logreg, Xva, yva, setting_res)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300,
            early_stopping=True, random_state=SEED,
        )
        mlp.fit(Xtr, ytr)
        eval_model(f"{setting} / mlp", mlp, Xva, yva, setting_res)

        # leakage diagnostic (logreg): macro F1 on report-derived vs ICD-only val rows
        by_prov = {}
        for tag in ["report", "icd_only"]:
            mask = val_prov == tag
            if mask.sum() > 0:
                by_prov[tag] = {
                    "n": int(mask.sum()),
                    "macro_f1": round(float(f1_score(yva[mask], lr_pred[mask], average="macro")), 4),
                }
        setting_res["logreg_macro_f1_by_label_provenance"] = by_prov
        print(f"\n  [{setting}] logreg macro F1 by label provenance: {by_prov}")

        results["settings"][setting] = setting_res

        joblib.dump(logreg, OUT_DIR / f"logreg_{setting}.joblib")
        proba = logreg.predict_proba(Xva)
        for i, n in enumerate(LABEL_NAMES):
            val_pred_export[f"text_{setting}_p_{n.replace(' ', '_').lower()}"] = proba[:, i]

    val_pred_export.to_csv(OUT_DIR / "val_predictions.csv", index=False)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    a = results["settings"]["indication"]
    b = results["settings"]["full_report"]
    a_lr = a["indication / logreg"]
    b_lr = b["full_report / logreg"]
    a_prov = a["logreg_macro_f1_by_label_provenance"]
    b_prov = b["logreg_macro_f1_by_label_provenance"]
    lines = [
        "# Text unimodal baseline -- results",
        "",
        f"Encoder: frozen {MODEL_NAME}, mean-pooled. Head: multinomial logistic regression "
        "(class_weight balanced). Metric: macro F1 on val (test untouched).",
        "",
        "## Setting A -- `indication` section only  (THE honest number)",
        "",
        "| head | macro F1 | Normal F1 | Pneumonia F1 | Cancer F1 |",
        "|---|---|---|---|---|",
        f"| logreg | {a_lr['macro_f1']} | {a_lr['f1_per_class']['Normal']} | "
        f"{a_lr['f1_per_class']['Pneumonia']} | {a_lr['f1_per_class']['Lung Cancer']} |",
        f"| mlp | {a['indication / mlp']['macro_f1']} | "
        f"{a['indication / mlp']['f1_per_class']['Normal']} | "
        f"{a['indication / mlp']['f1_per_class']['Pneumonia']} | "
        f"{a['indication / mlp']['f1_per_class']['Lung Cancer']} |",
        "",
        "## Setting B -- full report  (LEAKAGE-INFLATED, not a capability estimate)",
        "",
        f"logreg macro F1 {b_lr['macro_f1']} "
        f"(Normal {b_lr['f1_per_class']['Normal']} / Pneumonia {b_lr['f1_per_class']['Pneumonia']} "
        f"/ Cancer {b_lr['f1_per_class']['Lung Cancer']}). "
        "Labels were partly derived from this text; the score reflects that.",
        "",
        "## Leakage analysis -- macro F1 split by how the label was derived",
        "",
        "| val subset | n | Setting A (indication) | Setting B (full report) |",
        "|---|---|---|---|",
        f"| label had report-derived evidence | {a_prov['report']['n']} | "
        f"{a_prov['report']['macro_f1']} | {b_prov['report']['macro_f1']} |",
        f"| label is ICD-only (report independent) | {a_prov['icd_only']['n']} | "
        f"{a_prov['icd_only']['macro_f1']} | {b_prov['icd_only']['macro_f1']} |",
        "",
        "Reading: on the ICD-only subset (labels the report cannot leak), seeing the whole "
        f"report barely beats seeing only the indication ({b_prov['icd_only']['macro_f1']} vs "
        f"{a_prov['icd_only']['macro_f1']}) -- so Setting B's headline lead over Setting A "
        f"({b_lr['macro_f1']} vs {a_lr['macro_f1']}) is almost entirely the report-derived "
        f"subset ({b_prov['report']['macro_f1']} vs {a_prov['report']['macro_f1']}), i.e. "
        "leakage. Setting A is close to the true text ceiling. (The report vs ICD-only gap "
        "within Setting A also partly reflects task difficulty + class mix: the ICD-only "
        "subset is all Pneumonia/Cancer with no clear radiographic finding, and contains no "
        "Normal cases.)",
        "",
        "## Baseline comparison (val macro F1)",
        "- Image v3 (frozen MIMIC-CXR DenseNet): 0.50  -- strongest on Pneumonia (F1 0.77)",
        "- Blood (HistGradientBoosting): 0.46",
        f"- Text A (indication): {a_lr['macro_f1']}  -- strongest on Lung Cancer "
        f"(F1 {a_lr['f1_per_class']['Lung Cancer']}, vs image 0.17 / blood 0.20)",
        "The three modalities are complementary -> Phase 3 fusion has a real case.",
        "",
        "Artifacts: logreg_indication.joblib, logreg_full_report.joblib, "
        "val_predictions.csv (both settings' probabilities, for Phase 3 fusion), "
        "cached embeddings emb_*.npy, metrics.json",
    ]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved everything to {OUT_DIR}/")


if __name__ == "__main__":
    main()
