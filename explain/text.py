"""
Token attribution for the text model (Setting A -- INDICATION section only).

Pipeline being explained: indication text -> frozen Bio_ClinicalBERT (mean-pooled) ->
LogisticRegression (experiments/text_baseline/logreg_indication.joblib).

Method: SHAP with a Text masker. SHAP masks spans of the input, re-runs the full
embed+classify pipeline, and attributes the change to each token -- so it explains the
whole non-linear pipeline end to end, not just the linear head. Same SHAP framing as the
blood modality. Attribution is in probability space (the predict fn returns probabilities).

INDICATION extraction + cleaning mirror scripts/train_text_baseline.py (kept in sync by
hand; training code is intentionally not imported).

First run downloads Bio_ClinicalBERT (~430 MB) and is slow; then ~a few seconds/case on CPU.

CLI:  python -m explain.text --study-id 50296395
      python -m explain.text --text "65M smoker, chronic cough and weight loss"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import joblib
import shap
from transformers import AutoModel, AutoTokenizer

from explain import LABEL_NAMES

MODEL_PATH = Path("experiments/text_baseline/logreg_indication.joblib")
ENCODER = "emilyalsentzer/Bio_ClinicalBERT"
MAX_TOKENS = 512

INDICATION_HEADERS = [
    "INDICATION", "HISTORY", "CLINICAL HISTORY", "CLINICAL INDICATION",
    "CLINICAL INFORMATION", "REASON FOR EXAMINATION", "REASON FOR EXAM", "PATIENT HISTORY",
]
_HEADER_RE = r"\n\s*[A-Z][A-Z /()\-]{2,40}:"

_cache: dict = {}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("___", " ")).strip()


def extract_indication(report_text: str) -> str:
    for hdr in INDICATION_HEADERS:
        m = re.search(rf"{hdr}:\s*(.*?)(?={_HEADER_RE}|\Z)", report_text, re.S)
        if m and m.group(1).strip():
            frag = m.group(1)
            frag = frag.split("//")[0] if "//" in frag else frag
            return clean(frag)
    return ""


def _load():
    if not _cache:
        tok = AutoTokenizer.from_pretrained(ENCODER)
        enc = AutoModel.from_pretrained(ENCODER).eval()
        for p in enc.parameters():
            p.requires_grad = False
        _cache.update(tok=tok, enc=enc, head=joblib.load(MODEL_PATH))
    return _cache["tok"], _cache["enc"], _cache["head"]


@torch.no_grad()
def _embed(texts: list[str]) -> np.ndarray:
    tok, enc, _ = _load()
    e = tok(list(texts), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    h = enc(**e).last_hidden_state
    mask = e["attention_mask"].unsqueeze(-1).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled.numpy().astype(np.float32)


def predict_proba_texts(texts) -> np.ndarray:
    texts = [t if t else "[CLS]" for t in np.atleast_1d(texts)]
    _, _, head = _load()
    return head.predict_proba(_embed(texts))


def explain_text(indication_text: str, top_k: int = 10) -> dict:
    text = clean(indication_text)
    proba = predict_proba_texts([text])[0]
    pred = int(proba.argmax())

    # word-level masker (\W+ split) -> clean whole-word attributions, not BERT subwords
    explainer = shap.Explainer(predict_proba_texts, shap.maskers.Text(r"\W+"),
                               output_names=LABEL_NAMES)
    sv = explainer([text], max_evals=300, silent=True)

    tokens = [t.strip() for t in sv.data[0]]
    effects = np.asarray(sv.values[0])[:, pred]            # per-word effect on P(pred)
    keep = [i for i, t in enumerate(tokens) if t != ""]
    order = sorted(keep, key=lambda i: -abs(effects[i]))

    items = [
        {
            "feature": tokens[i],
            "value": None,
            "effect": round(float(effects[i]), 4),
            "direction": "supports" if effects[i] > 0 else "against",
        }
        for i in order[:top_k]
    ]
    supports = [it["feature"] for it in items if it["direction"] == "supports"][:4]
    summary = (
        f"Text model predicts {LABEL_NAMES[pred]} (p={proba[pred]:.2f}) from the indication. "
        + (f"Most influential words: {', '.join(repr(w) for w in supports)}." if supports else "")
    )
    return {
        "modality": "text",
        "predicted_class": LABEL_NAMES[pred],
        "probabilities": {n: round(float(p), 4) for n, p in zip(LABEL_NAMES, proba)},
        "attributions": {"method": "SHAP (Text masker, probability)", "items": items},
        "summary": summary,
        "_raw": {"tokens": tokens, "effect_per_class": np.asarray(sv.values[0]).tolist(),
                 "input_text": text},
    }


def plot_text(res: dict, out_path: str | Path) -> Path:
    """Render the indication with each word shaded by its effect on the predicted class.
    `res` is the dict returned by explain_text()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    tokens = res["_raw"]["tokens"]
    pred_i = LABEL_NAMES.index(res["predicted_class"])
    eff = np.asarray(res["_raw"]["effect_per_class"])[:, pred_i]
    vmax = max(np.abs(eff).max(), 1e-6)

    fig, ax = plt.subplots(figsize=(9, 1.6))
    ax.axis("off")
    x = 0.01
    for t, e in zip(tokens, eff):
        if t == "":
            continue
        frac = e / vmax
        color = (0.75, 0.20, 0.15, min(abs(frac), 1)) if e > 0 else (0.17, 0.50, 0.72, min(abs(frac), 1))
        w = 0.011 * (len(t) + 1.5)
        ax.add_patch(FancyBboxPatch((x, 0.4), w, 0.32, boxstyle="round,pad=0.004",
                                    fc=color, ec="none", transform=ax.transAxes))
        ax.text(x + w / 2, 0.56, t, ha="center", va="center", fontsize=10, transform=ax.transAxes)
        x += w + 0.004
        if x > 0.96:
            x = 0.01
    ax.set_title(res["summary"], fontsize=9, loc="left", wrap=True)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _indication_from_split(study_id: int, split: str = "val") -> tuple[str, str]:
    import pandas as pd
    df = pd.read_csv(f"data/processed/{split}.csv")
    row = df.loc[df["study_id"] == study_id]
    if row.empty:
        raise SystemExit(f"study_id {study_id} not in {split}.csv")
    r = row.iloc[0]
    report = Path(r["report_path"]).read_text(encoding="utf-8", errors="ignore")
    return extract_indication(report), r["final_label"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-id", type=int)
    ap.add_argument("--text", type=str)
    ap.add_argument("--split", default="val")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    if args.text:
        indication, true_label = args.text, "(n/a)"
    elif args.study_id:
        indication, true_label = _indication_from_split(args.study_id, args.split)
    else:
        raise SystemExit("pass --study-id or --text")

    print(f"indication: {indication!r}\ntrue label: {true_label}")
    res = explain_text(indication)
    print(f"probabilities: {res['probabilities']}\n\n{res['summary']}\n")
    print(f"{'token':<18}{'effect':>10}  direction")
    for it in res["attributions"]["items"]:
        print(f"{it['feature']:<18}{it['effect']:>10}  {it['direction']}")
    if args.plot:
        print(f"\nsaved plot -> {plot_text(indication, args.plot)}")
