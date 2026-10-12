"""
Faithfulness check for the LLM-written background sentences. A sentence is shown only if it
passes all three tests against the one passage it cites; otherwise it is dropped and counted.

  1. Entailment: an NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) scores whether the
     passage entails the sentence. The passage is cut into windows of WINDOW consecutive
     sentences and the best window counts, because the NLI model reads about 512 tokens and a
     claim usually rests on two or three neighbouring sentences.
  2. Numbers: every number in the sentence must appear in the passage. Catches a changed
     figure. It cannot catch a true figure attached to the wrong claim (a mortality rate for
     pneumonia attributed to a biomarker); only the entailment test can.
  3. Wording: no treatment or advice wording, and no certainty words ("causes", "proves"). The
     passages talk about association, and the tool must never sound like it gives advice.

Policy is "verify then show": a failed sentence is removed, not flagged and kept. The result
reports how many of the written sentences passed, so a weak writer shows up as a low rate.

The entailment threshold is set on the hand-made set in eval_faithfulness.py: every unfaithful
sentence scored 0.19 or lower and every faithful one 0.48 or higher, so 0.30 sits in the gap. That
set is small (35 sentences) and written by the project author, so the threshold is a reasonable
setting, not a proven one. The wording test is deliberately blunt: it also drops a faithful
sentence that merely reports a paper's own mention of "treatment".
"""
from __future__ import annotations

import re

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
WINDOW = 3
MAX_WINDOWS = 5       # score only the windows sharing the most words with the sentence (speed)
ENTAIL_MIN = 0.30          # inside the gap between the hand-made set's classes, see eval_faithfulness.py

# (?<!-) and (?!-): a hyphen is a regex word boundary, so without them "all-cause mortality" and
# "cause-specific" (standard epidemiology terms, not causal claims) were dropped as "causes".
BANNED = re.compile(r"(?<!-)\b(treat\w*|therap\w*|dos(e|es|age)|prescrib\w*|recommend\w*|should|advis\w*|"
                    r"must|medicat\w*|causes?|caused|proves?|proven|confirms?|guarantee\w*)\b(?!-)", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")

_nli = None


def _model():
    global _nli
    if _nli is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        entail = next(i for i, name in mdl.config.id2label.items() if name.lower().startswith("entail"))
        _nli = (tok, mdl, entail, torch)
    return _nli


def _windows(passage: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", passage) if s.strip()]
    if len(sents) <= WINDOW:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + WINDOW]) for i in range(len(sents) - WINDOW + 1)]


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


def entailment(sentence: str, passage: str) -> float:
    """Best entailment over the passage windows most related to the sentence. Scoring all windows of a
    300-word passage took about 15 NLI passes per sentence on CPU; the claim can only rest on windows
    that share its words, so the rest are skipped."""
    tok, mdl, entail, torch = _model()
    wins = _windows(passage)
    if len(wins) > MAX_WINDOWS:
        sw = _content_words(sentence)
        wins = sorted(wins, key=lambda w: -len(sw & _content_words(w)))[:MAX_WINDOWS]
    enc = tok(wins, [sentence] * len(wins), return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[:, entail]
    return float(probs.max())


def _numbers(text: str) -> set[str]:
    return set(_NUM.findall(text.replace("^", "").replace(",", "")))


def check_sentence(sentence: str, passage: str) -> dict:
    ent = entailment(sentence, passage)
    extra = sorted(_numbers(sentence) - _numbers(passage))
    banned = sorted({m.group(0).lower() for m in BANNED.finditer(sentence)})
    reasons = []
    if ent < ENTAIL_MIN:
        reasons.append(f"entailment {ent:.2f} below {ENTAIL_MIN}")
    if extra:
        reasons.append("numbers not in passage: " + ", ".join(extra))
    if banned:
        reasons.append("banned wording: " + ", ".join(banned))
    return {"entailment": round(ent, 3), "numbers_not_in_passage": extra, "banned": banned,
            "passed": not reasons, "reasons": reasons}


def verify(explanation: dict, evidence: dict) -> dict:
    """Checks each background sentence against the passage it cites. Returns the explanation with
    `verified` (shown), `dropped` (removed, with reasons) and a pass rate."""
    by_n = {p["n"]: p for p in evidence["passages"]}
    verified, dropped = [], []
    for b in explanation["background"]:
        p = by_n.get(b["n"])
        if p is None:
            dropped.append({**b, "reasons": ["cites a passage that does not exist"]})
            continue
        r = check_sentence(b["text"], p["text"])
        (verified if r["passed"] else dropped).append({**b, **r})
    n = len(explanation["background"])
    return {**explanation, "verified": verified, "dropped": dropped,
            "pass_rate": (len(verified) / n) if n else None}
