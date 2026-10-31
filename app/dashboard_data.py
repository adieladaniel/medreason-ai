"""
Builds the consolidated data file behind the progress dashboard (app/data/dashboard.json).

Everything is read from the real result files, not typed in by hand, so the dashboard
updates when a new experiment is added: experiments/*/metrics.json and *.csv, the val
prediction files (for ROC, precision-recall, calibration, and modality-agreement
analysis), the Kaggle training log (for the image training curve), the pipeline logs
in scripts/, the split CSVs, and app/data/phases.json (for phase status).

A small number of figures exist only in the docs, not in a machine-readable file: image
baselines v1 and v2 (no run logs were kept). Those are listed in IMAGE_VERSIONS below with
the source noted.

The output contains aggregates only. It carries no study, subject, or admission IDs, so it
is safe to commit. Sections whose source files are missing are set to None and the UI
shows "not available in this checkout".

Run directly (python -m app.dashboard_data) or let app.main rebuild it on startup when a
source file is newer than the output.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "app" / "data" / "dashboard.json"

CLASSES = ["Normal", "Pneumonia", "Lung Cancer"]
CLS = ["normal", "pneumonia", "lung_cancer"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Image baselines v1 and v2 have no saved logs. The numbers were written down when those runs finished (Phase 2).
# v3 is the Kaggle run whose log is parsed below.
IMAGE_VERSIONS = [
    {
        "key": "v1", "label": "v1",
        "desc": "Frozen ImageNet DenseNet121, linear head, 8 epochs",
        "macro_f1": 0.45, "f1": {"Normal": 0.49, "Pneumonia": 0.73, "Lung Cancer": 0.12},
        "source": "Phase 2 notes (no run log was kept)",
    },
    {
        "key": "v2", "label": "v2",
        "desc": "Last dense block unfrozen, discriminative learning rate, 20 epochs",
        "macro_f1": 0.46, "f1": {"Normal": 0.52, "Pneumonia": 0.71, "Lung Cancer": 0.14},
        "source": "Phase 2 notes (no run log was kept)",
    },
    {
        "key": "v3", "label": "v3",
        "desc": "Frozen MIMIC-CXR DenseNet121 (torchxrayvision), 18 pathology outputs (11 trained), MLP head, 15 epochs",
        "macro_f1": 0.50, "f1": {"Normal": 0.56, "Pneumonia": 0.77, "Lung Cancer": 0.17},
        "source": "Kaggle run log",
    },
]

PHASES_FILE = "app/data/phases.json"

# Work that is not done. Every item changes the models or needs data or compute that this project did not have, and the sealed test
# split is spent, so any new test number would be post-hoc.
REMAINING = [
    {"phase": "An explicit \"other\" outcome", "work": "Learn features for studies outside the three classes, for example by fine-tuning the image model on finding labels taken from the reports. A gate on the existing outputs and a retrain on frozen features both failed their pre-registered rules.", "effort": "GPU training"},
    {"phase": "A stronger image model", "work": "Replace or fine-tune the frozen image backbone; the image adds almost nothing to the fused result on the sealed test split.", "effort": "GPU training"},
    {"phase": "External validation", "work": "Test at a second site or on a public set such as NIH ChestX-ray14.", "effort": "another dataset"},
    {"phase": "Faster sentence checker", "work": "Run the entailment checker on a GPU or replace it with a smaller model; it is the largest part of the wait for a cited note.", "effort": "CUDA build or a smaller model"},
]

FUSION_LABELS = {
    "logpool": ("Geometric mean (log pool)", "pooled"),
    "mean_probability": ("Arithmetic mean", "pooled"),
    "logreg_stack": ("Logistic stack", "trained"),
    "logreg_stack_balanced": ("Logistic stack, class-balanced", "trained"),
    "hgb_stack": ("Gradient-boosted stack", "trained"),
}

BLOOD_TIERS = {
    "WBC": "Tier 1 core", "RBC": "Tier 1 core", "Hemoglobin": "Tier 1 core",
    "Hematocrit": "Tier 1 core", "Platelets": "Tier 1 core",
    "Neutrophils": "Tier 1 differential", "Lymphocytes": "Tier 1 differential",
    "Monocytes": "Tier 1 differential",
    "CRP": "Tier 2", "ESR": "Tier 2", "Albumin": "Tier 2", "LDH": "Tier 2",
    "Sodium": "Tier 3", "Potassium": "Tier 3", "Glucose": "Tier 3",
    "Creatinine": "Tier 3", "Urea": "Tier 3",
}

WATCHED = [
    "app/data/phases.json",
    "experiments/*/metrics.json",
    "experiments/*/val_predictions.csv",
    "experiments/*/feature_importance.csv",
    "experiments/fusion/logreg_stack_coef.csv",
    "kaggle_upload/baseline_kernel/output/*.log",
    "experiments/rag_retrieval/*.json",
    "experiments/out_of_scope/results.json",
    "experiments/v2_four_class/results.json",
]

FUSION_MLP_LABELS = {"image": "Image", "blood": "Blood", "text": "Text"}


def _r(x, d=4):
    return None if x is None else round(float(x), d)


def _json(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _text(rel: str):
    p = ROOT / rel
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None


def _safe(fn):
    """Run a section builder; return None if its source files are missing or malformed."""
    try:
        return fn()
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------- phases
def _phases():
    """Phase status is data, kept in a small tracked file, so the dashboard does not depend on any document."""
    return _json(PHASES_FILE)


# ---------------------------------------------------------------- data
def _grab(pattern: str, text: str):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _data():
    linking = _text("scripts/exploration/linking_result.log") or ""
    funnel = {
        "studies": _grab(r"Total CXR studies \(distinct study_id\): (\d+)", linking),
        "linked_studies": _grab(r"Studies with >=1 matching admission: (\d+)", linking),
        "subjects": _grab(r"Total CXR subjects \(distinct subject_id\): (\d+)", linking),
        "linked_subjects": _grab(r"Subjects with >=1 matching admission: (\d+)", linking),
        "ambiguous": _grab(r"ambiguous\): (\d+)", linking),
        "admissions": _grab(r"Distinct hadm_id resolved: (\d+)", linking),
    }

    splits = {}
    for s in ("train", "val", "test"):
        p = ROOT / "data" / "processed" / f"{s}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["subject_id", "final_label"])
        vc = df["final_label"].value_counts()
        splits[s] = {
            "n": int(len(df)),
            "subjects": int(df["subject_id"].nunique()),
            "counts": {c: int(vc.get(c, 0)) for c in CLASSES},
        }
    funnel["labeled"] = sum(v["n"] for v in splits.values()) or None

    coverage = []
    cov_text = _text("scripts/exploration/blood_coverage.log") or ""
    for m in re.finditer(r"^(\w+)\s+\d+\s+\d+\s+([\d.]+)%", cov_text, re.M):
        name = m.group(1)
        if name in BLOOD_TIERS:
            coverage.append({"test": name, "tier": BLOOD_TIERS[name], "pct": float(m.group(2))})

    provenance = None
    excluded = None
    p = ROOT / "data" / "interim" / "final_study_labels.csv"
    if p.exists():
        f = pd.read_csv(p)
        excluded = int((f["final_label"] == "Excluded").sum())
        pn = f[f["final_label"] == "Pneumonia"]
        ca = f[f["final_label"] == "Lung Cancer"]
        nm = f[f["final_label"] == "Normal"]
        pm = pn["pneumonia_mentioned"].astype(bool)
        cm = ca["cancer_mentioned"].astype(bool)
        provenance = {
            "Pneumonia": {
                "icd_only": int(((pn["icd_pneumonia"] == 1) & ~pm).sum()),
                "report_only": int(((pn["icd_pneumonia"] == 0) & pm).sum()),
                "both": int(((pn["icd_pneumonia"] == 1) & pm).sum()),
            },
            "Lung Cancer": {
                "icd_only": int(((ca["icd_cancer"] == 1) & ~cm).sum()),
                "report_only": int(((ca["icd_cancer"] == 0) & cm).sum()),
                "both": int(((ca["icd_cancer"] == 1) & cm).sum()),
            },
            "Normal": {"report_clean": int(len(nm))},
        }

    reports = _text("scripts/exploration/label_reports.log") or ""
    mentions = {
        "no_finding": _grab(r"No Finding: (\d+)", reports),
        "pneumonia": _grab(r"Pneumonia mentioned: (\d+)", reports),
        "cancer": _grab(r"Cancer mentioned: (\d+)", reports),
    }
    return {
        "funnel": funnel, "splits": splits, "blood_coverage": coverage,
        "provenance": provenance, "excluded": excluded, "report_mentions": mentions,
    }


# ---------------------------------------------------------------- models
def _image_curve():
    p = ROOT / "kaggle_upload" / "baseline_kernel" / "output" / "medreason-ai-baseline-image-classifier.log"
    if not p.exists():
        return None
    events = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for e in events:
        m = re.search(r"Epoch (\d+)/\d+ - train_loss: ([\d.]+) - val_macro_f1: ([\d.]+)", e.get("data", ""))
        if m:
            rows.append({"epoch": int(m.group(1)), "train_loss": float(m.group(2)),
                         "val_macro_f1": float(m.group(3))})
    return rows or None


def _models():
    fusion = _json("experiments/fusion/metrics.json")
    curve = _image_curve()
    image = {
        "versions": IMAGE_VERSIONS,
        "training_curve": curve,
        "best_epoch": max(curve, key=lambda r: r["val_macro_f1"])["epoch"] if curve else None,
        "fusion_input_variant": (fusion or {}).get("single_modality_val", {}).get("image"),
    }

    blood = None
    bm = _json("experiments/blood_baseline/metrics.json")
    if bm:
        labels = {
            "dummy_most_frequent": "Majority class",
            "logreg": "Logistic regression",
            "hist_gradient_boosting": "Gradient-boosted trees",
        }
        imp = []
        p = ROOT / "experiments" / "blood_baseline" / "feature_importance.csv"
        if p.exists():
            df = pd.read_csv(p)
            imp = [{"feature": r.feature, "mean": _r(r.importance_mean, 5), "std": _r(r.importance_std, 5)}
                   for r in df.itertuples()]
        h = bm["hist_gradient_boosting"]
        blood = {
            "models": [
                {"key": k, "label": labels[k], "macro_f1": bm[k]["macro_f1"],
                 "balanced_accuracy": bm[k]["balanced_accuracy"], "f1": bm[k]["f1_per_class"]}
                for k in labels
            ],
            "importance": imp,
            "core_lab_rows_f1": h.get("macro_f1_rows_with_core_lab"),
            "core_lab_rows_n": h.get("val_rows_with_core_lab"),
            "features": len(bm.get("features", [])),
        }

    text = None
    tm = _json("experiments/text_baseline/metrics.json")
    if tm:
        def pack(setting):
            s = tm["settings"][setting]
            return {
                "logreg": s[f"{setting} / logreg"],
                "mlp": s[f"{setting} / mlp"],
                "by_provenance": s["logreg_macro_f1_by_label_provenance"],
            }
        text = {"encoder": tm.get("encoder"), "indication": pack("indication"),
                "full_report": pack("full_report")}
    return {"image": image, "blood": blood, "text": text}


# ---------------------------------------------------------------- fusion + predictions
def _fusion():
    fm = _json("experiments/fusion/metrics.json")
    if not fm:
        return None
    methods = []
    for k, v in fm["fusion_val"].items():
        key = k.split(" / ")[1]
        label, kind = FUSION_LABELS.get(key, (key, "trained"))
        methods.append({"key": key, "label": label, "kind": kind, "macro_f1": v["macro_f1"],
                        "f1": v["f1_per_class"], "confusion": v["confusion_matrix"]})
    methods.sort(key=lambda m: -m["macro_f1"])

    coef = None
    p = ROOT / "experiments" / "fusion" / "logreg_stack_coef.csv"
    if p.exists():
        df = pd.read_csv(p, index_col=0)
        coef = {"rows": list(df.index), "cols": list(df.columns),
                "values": df.round(3).values.tolist()}
    return {
        "single": fm["single_modality_val"], "methods": methods,
        "ablation": fm["ablation_val_macro_f1"], "coef": coef,
        "n_train": fm["n_train"], "n_val": fm["n_val"],
    }


def _val_predictions():
    sources = {
        "image": ("experiments/image_baseline/val_predictions.csv", [f"image_p_{c}" for c in CLS]),
        "blood": ("experiments/blood_baseline/val_predictions.csv", [f"blood_p_{c}" for c in CLS]),
        "text": ("experiments/text_baseline/val_predictions.csv", [f"text_indication_p_{c}" for c in CLS]),
        "fusion": ("experiments/fusion/val_predictions.csv", [f"fused_p_{c}" for c in CLS]),
    }
    frames, y = {}, None
    for name, (rel, cols) in sources.items():
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(rel)
        df = pd.read_csv(p).set_index("study_id")
        frames[name] = df[cols]
        if name == "fusion":
            y = df["final_label"]
    common = frames["fusion"].index
    for f in frames.values():
        common = common.intersection(f.index)
    P = {n: f.loc[common].to_numpy() for n, f in frames.items()}
    yy = y.loc[common].map(LABEL_TO_IDX).to_numpy()
    return P, yy


def _curves(P, y):
    grid = np.linspace(0, 1, 101)
    out = {}
    for name, p in P.items():
        roc, pr = {}, {}
        for k, c in enumerate(CLASSES):
            yk = (y == k).astype(int)
            fpr, tpr, _ = roc_curve(yk, p[:, k])
            roc[c] = {"auc": _r(roc_auc_score(yk, p[:, k])),
                      "tpr": np.interp(grid, fpr, tpr).round(4).tolist()}
            prec, rec, _ = precision_recall_curve(yk, p[:, k])
            order = np.argsort(rec)
            pr[c] = {"ap": _r(average_precision_score(yk, p[:, k])),
                     "prevalence": _r(yk.mean()),
                     "precision": np.interp(grid, rec[order], prec[order]).round(4).tolist()}
        conf, correct = p.max(1), p.argmax(1) == y
        edges = np.linspace(1 / 3, 1.0, 8)
        idx = np.clip(np.digitize(conf, edges[1:-1]), 0, 6)
        bins, ece = [], 0.0
        for b in range(7):
            m = idx == b
            if m.sum() == 0:
                continue
            bins.append({"conf": _r(conf[m].mean()), "acc": _r(correct[m].mean()), "n": int(m.sum())})
            ece += m.sum() / len(y) * abs(correct[m].mean() - conf[m].mean())
        out[name] = {"roc": roc, "pr": pr, "calibration": {"bins": bins, "ece": _r(ece)}}
    return {"grid": grid.round(3).tolist(), "models": out}


def _agreement(P, y):
    mods = ["image", "blood", "text"]
    preds = np.stack([P[m].argmax(1) for m in mods], axis=1)
    fused = P["fusion"].argmax(1)
    top_agree = np.array([np.bincount(r, minlength=3).max() for r in preds])
    single_acc = np.stack([(preds[:, i] == y) for i in range(3)], axis=1)
    levels = []
    for lvl, name in ((3, "All three agree"), (2, "Two agree"), (1, "All three differ")):
        m = top_agree == lvl
        if m.sum() == 0:
            continue
        levels.append({
            "name": name, "n": int(m.sum()), "share": _r(m.mean()),
            "fused_accuracy": _r((fused[m] == y[m]).mean()),
            "mean_single_accuracy": _r(single_acc[m].mean()),
        })
    n_correct = single_acc.sum(axis=1)
    per_class = []
    for k, c in enumerate(CLASSES):
        m = y == k
        per_class.append({
            "class": c, "n": int(m.sum()),
            "dist": [_r((n_correct[m] == j).mean()) for j in range(4)],
            "fused_correct": _r((fused[m] == k).mean()),
        })
    return {"levels": levels, "per_class": per_class}


def _predictions_section():
    P, y = _val_predictions()
    return {"curves": _curves(P, y), "agreement": _agreement(P, y), "n": int(len(y))}


# ---------------------------------------------------------------- feature-level fusion ablation
def _fusion_mlp():
    m = _json("experiments/fusion_mlp/metrics.json")
    if not m:
        return None
    ablation = {
        "+".join(FUSION_MLP_LABELS[p] for p in k.split("+")): {"dev": v, "val": m["ablation_val_macro_f1"][k]}
        for k, v in m["ablation_dev_macro_f1"].items()
    }
    return {
        "late_fusion_bar": m["late_fusion_bar"],
        "final": m["final"],
        "pca_scan": m["pca_scan_dev_macro_f1"],
        "text_pca_dim": m["text_pca_dim"],
        "n_fit": m["n_fit"], "n_dev": m["n_dev"], "n_val": m["n_val"],
        "ablation": ablation,
        "verdict": m["verdict"],
    }


# ---------------------------------------------------------------- phase 6, evidence retrieval
RAG_VARIANTS = [   # (key in pipeline.json, label shown), in the order the story is told
    ("dense (baseline)", "Embeddings only (first pick)"),
    ("keyword only (BM25)", "Keyword search only"),
    ("hybrid", "Embeddings + keywords"),
    ("hybrid + MedCPT rerank (top 5)", "Hybrid + reranker (adopted)"),
]


def _rag():
    pipe = _json("experiments/rag_retrieval/pipeline.json")
    if not pipe:
        return None
    sources = _json("app/rag/corpus_sources.json") or {}
    out = {
        "papers": len(sources.get("sources", [])),
        "random_p1": pipe["random"]["p@1"],
        "variants": [{"label": lab, "dev_p1": pipe[k]["dev"]["p@1"], "test_p1": pipe[k]["test"]["p@1"],
                      "mrr": pipe[k]["all"]["mrr"], "abstain_auc": pipe[k].get("abstain_auc_rerank")}
                     for k, lab in RAG_VARIANTS if k in pipe],
        "n_dev": pipe["hybrid"]["dev"]["n"], "n_test": pipe["hybrid"]["test"]["n"],
    }
    faith = _json("experiments/rag_retrieval/faithfulness_eval.json")
    if faith:
        good = [r["entailment"] for r in faith if r["faithful"]]
        bad = [r["entailment"] for r in faith if not r["faithful"]]
        out["faithfulness"] = {"n_faithful": len(good), "n_unfaithful": len(bad),
                               "min_faithful": min(good), "max_unfaithful": max(bad),
                               "points": [{"entailment": r["entailment"], "faithful": r["faithful"]} for r in faith]}
    stages = [("27 papers", "gap_probe_27papers.json"), ("44 papers", "gap_probe_44papers_nofallback.json"),
              ("44 papers, fallback phrasing", "gap_probe.json")]
    cov = []
    for label, fn in stages:
        rows = _json(f"experiments/rag_retrieval/{fn}")
        if rows:
            cov.append({"stage": label, "queries": len(rows), "gaps": sum(1 for r in rows if r["on_topic_hits"] == 0)})
    if cov:
        out["coverage"] = cov
    checkers = _json("experiments/rag_retrieval/checker_eval.json")
    if checkers:
        label = {"deberta-nli": "DeBERTa NLI (current)", "deberta-v3-large": "MiniCheck DeBERTa-v3-Large", "flan-t5-large": "MiniCheck Flan-T5-Large"}
        out["checkers"] = [{"name": label.get(k, k), "threshold": v["threshold"], "seconds": v["seconds"],
                            "hand": {x: v["hand"][x] for x in ("auroc", "false_reject_n", "faithful", "caught_n", "unfaithful")},
                            "real": {x: v["real"][x] for x in ("auroc", "false_reject_n", "faithful", "caught_n", "unfaithful")}}
                           for k, v in checkers.items()]
    writer = _json("experiments/rag_retrieval/writer_eval.json")
    if writer:
        out["writer"] = [{k: w[k] for k in ("model", "expand", "n", "written", "pass", "entail", "words", "seconds")}
                         for w in writer]
    return out


# ---------------------------------------------------------------- reliability (uncertainty, calibration, subgroups)
def _uncertainty():
    """experiments/uncertainty/metrics.json holds aggregates only (no identifiers), so it passes through."""
    return _json("experiments/uncertainty/metrics.json")


# ---------------------------------------------------------------- can the system say "none of these"?
def _scope():
    """Two attempts to flag studies outside Normal, Pneumonia and Lung Cancer, scored on the same patients: in-scope validation studies against the
    Excluded studies of patients in no split. Aggregates only. Both were pre-registered and neither met its adoption rule."""
    gate = _json("experiments/out_of_scope/results.json")["final_held_out_patients"]
    v2 = _json("experiments/v2_four_class/results.json")
    s = v2["statistics"]
    rows = [
        {"label": "Existing confidence signal (1 minus the top probability)", "auroc": gate["point"]["auroc_existing_confidence"],
         "low": gate["interval"]["auroc_existing_confidence"][0], "high": gate["interval"]["auroc_existing_confidence"][1],
         "flagged_other": None, "flagged_in_scope": None, "adopted": None},
        {"label": "Gate on the models' outputs and the image backbone's 18 outputs", "auroc": gate["point"]["auroc"],
         "low": gate["interval"]["auroc"][0], "high": gate["interval"]["auroc"][1], "flagged_other": gate["point"]["catch_rate"],
         "flagged_in_scope": gate["point"]["false_flag_rate"], "adopted": False},
        {"label": "Version 2: a binary other-versus-in-scope model per input, trained on the Excluded studies", "auroc": s["auroc"]["value"],
         "low": s["auroc"]["low"], "high": s["auroc"]["high"], "flagged_other": s["flagged_heldout"]["value"],
         "flagged_in_scope": s["flagged_inscope"]["value"], "adopted": bool(v2["adopted"])},
    ]
    return {"rows": rows, "required_auroc": v2["limits"]["auroc"], "required_flagged_other": v2["limits"]["flagged_heldout"],
            "allowed_flagged_in_scope": v2["limits"]["flagged_inscope"], "held_out": v2["sizes"]["heldout"],
            "in_scope": v2["sizes"]["in_scope_validation"], "conditions": v2["conditions"]}


# ---------------------------------------------------------------- assemble
def build() -> dict:
    data = _safe(_data)
    models = _safe(_models)
    fusion = _safe(_fusion)
    fusion_mlp = _safe(_fusion_mlp)
    preds = _safe(_predictions_section)
    rag = _safe(_rag)
    unc = _safe(_uncertainty)
    scope = _safe(_scope)

    head = {}
    if models and models["blood"]:
        head["blood"] = models["blood"]["models"][-1]["macro_f1"]
    if models and models["text"]:
        head["text"] = models["text"]["indication"]["logreg"]["macro_f1"]
    head["image"] = IMAGE_VERSIONS[-1]["macro_f1"]
    if fusion:
        best = fusion["methods"][0]
        head["fusion"] = best["macro_f1"]
        head["fusion_method"] = best["label"]
        head["best_single"] = max(v["macro_f1"] for v in fusion["single"].values())
    if data and data["funnel"].get("labeled"):
        head["total_samples"] = data["funnel"]["labeled"]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phases": _safe(_phases),
        "headline": head,
        "data": data,
        "models": models,
        "fusion": fusion,
        "fusion_mlp": fusion_mlp,
        "predictions": preds,
        "rag": rag,
        "uncertainty": unc,
        "scope": scope,
        "remaining": REMAINING,
    }


def write() -> dict:
    payload = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                        encoding="utf-8")
    return payload


def is_stale() -> bool:
    if not OUT_PATH.exists():
        return True
    built = OUT_PATH.stat().st_mtime
    for pattern in WATCHED:
        for p in ROOT.glob(pattern):
            if p.stat().st_mtime > built:
                return True
    return False


def keep_existing(payload: dict) -> dict:
    """A section whose source files are absent (a fresh clone has no data/ and no per-study predictions) comes back None. Keep the
    committed section instead of replacing it with nothing, so the dashboard of a clone shows what the repository ships."""
    if not OUT_PATH.exists():
        return payload
    try:
        old = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return payload
    return {k: (old[k] if v is None and old.get(k) is not None else v) for k, v in payload.items()}


def ensure_built() -> None:
    """Rebuild if the output is missing or a source is newer. Keep the old file if the
    sources are unavailable (for example on a fresh clone without the experiments)."""
    if not is_stale():
        return
    try:
        payload = build()
    except Exception:
        return
    if payload.get("models") or payload.get("fusion") or not OUT_PATH.exists():
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(keep_existing(payload), separators=(",", ":"), ensure_ascii=False),
                            encoding="utf-8")


if __name__ == "__main__":
    out = write()
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    for k in ("phases", "data", "models", "fusion", "predictions", "rag"):
        print(f"  {k}: {'ok' if out.get(k) else 'MISSING'}")
    print("  headline:", out["headline"])
