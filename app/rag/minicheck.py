"""
MiniCheck scorer (Tang, Laban and Durrett, EMNLP 2024), reimplemented from the reference code at
github.com/Liyan06/MiniCheck so the project does not depend on that package (its install failed to
build here). It answers one question: does this document support this claim? The score is the
probability of "supported", and the paper's default decision is score above 0.5.

Procedure, as in the reference inference code:
  * the document is cut into chunks of whole sentences, about 500 words for Flan-T5 and 400
    tokens for the RoBERTa and DeBERTa variants, and the best chunk counts (max over chunks);
  * input is "<chunk><eos><claim>", with the prefix "predict: " for Flan-T5;
  * Flan-T5 reads the logits of two vocabulary ids (3 for unsupported, 209 for supported) at the
    first decoder step; the classifier variants take softmax over two labels, label 1 = supported.

Variants: "flan-t5-large" (770M, MIT licence), "deberta-v3-large" and "roberta-large" (about 0.4B).
CPU only here, since the local torch build has no CUDA.
"""
from __future__ import annotations

import re

_MODELS = {"flan-t5-large": "lytang/MiniCheck-Flan-T5-Large",
           "deberta-v3-large": "lytang/MiniCheck-DeBERTa-v3-Large",
           "roberta-large": "lytang/MiniCheck-RoBERTa-Large"}
_CHUNK = {"flan-t5-large": 500, "deberta-v3-large": 400, "roberta-large": 400}
_MAX_LEN = {"flan-t5-large": 2048, "deberta-v3-large": 2048, "roberta-large": 512}

_cache: dict = {}


def _load(variant: str):
    if variant not in _cache:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
        ckpt = _MODELS[variant]
        tok = AutoTokenizer.from_pretrained(ckpt)
        if variant == "flan-t5-large":
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2).eval()
        _cache[variant] = (tok, model, torch)
    return _cache[variant]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()] or [""]


def _chunks(variant: str, tok, sents: list[str]) -> list[str]:
    limit, out, cur, count = _CHUNK[variant], [], [], 0
    for s in sents:
        n = len(s.split()) if variant == "flan-t5-large" else len(tok(s, add_special_tokens=False)["input_ids"])
        if cur and count + n > limit:
            out.append(" ".join(cur))
            cur, count = [s], n
        else:
            cur.append(s)
            count += n
    if cur:
        out.append(" ".join(cur))
    return [c for c in out if c.strip()]


def support_probability(variant: str, passage: str, claim: str) -> float:
    tok, model, torch = _load(variant)
    texts = [tok.eos_token.join([c, claim]) for c in _chunks(variant, tok, _sentences(passage))]
    if variant == "flan-t5-large":
        texts = ["predict: " + t for t in texts]
    enc = tok(texts, max_length=_MAX_LEN[variant], truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        if variant == "flan-t5-large":
            dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec).logits.squeeze(1)
            probs = torch.softmax(logits[:, [3, 209]], dim=-1)
        else:
            probs = torch.softmax(model(**enc).logits, dim=-1)
    return float(probs[:, 1].max())
