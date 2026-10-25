"""
Explanation writer. Split in two so a small local model cannot get the case wrong:

  1. Case sentences are built by code from the agent's facts: the prediction and its
     probability, the agreement between modalities, and the drivers. The LLM never writes them,
     so it cannot misstate the label, invent a driver, or misreport which modalities agreed.
  2. Background sentences are written by the LLM, one per retrieved passage, each from a single
     passage. The model may answer NOT_ADDRESSED, which drops the passage. With no passages
     the model is not called at all, so "no evidence" cannot be turned into an invented citation.

Why this design: a first version gave llama3.2 (3B) the whole case and a rulebook. It cited
passages that did not exist, wrote "associated with [2]" with nothing after it, and misreported
which modalities agreed. A small model follows a narrow task far better than a long rulebook, and
a sentence tied to one passage is something faithfulness.py can check against that passage.

Plain urllib to Ollama's HTTP API (no new dependency), temperature 0 for repeatability.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

from app.config import get_settings

NUM_CTX = 4096

SYSTEM = ("You summarise one passage from a medical research paper in a single sentence. "
          "Use only facts stated in the passage and copy any numbers exactly. "
          "Say 'associated with', never 'causes' or 'proves'. "
          "Do not give advice, treatment, or a diagnosis. "
          "If the passage does not report anything about the topic, reply with exactly: NOT_ADDRESSED")


def case_sentences(agent: dict, plan: dict) -> list[str]:
    """Deterministic sentences about this case, built only from the agent's facts."""
    belief, cons = agent["belief"], agent["consistency"]
    pred, probs = belief["predicted_class"], belief["probabilities"]
    out = [f"The trained models predict {pred} with {probs[pred]:.0%} probability "
           f"({', '.join(f'{c} {p:.0%}' for c, p in probs.items() if c != pred)} for the other classes)."]
    if cons.get("detail"):
        out.append(cons["detail"].strip())
    drivers = [f"{q['driver']} ({q['modality']})" for q in plan["queries"] if q["modality"] != "agent"]
    if drivers:
        out.append("Findings that supported this prediction: " + "; ".join(drivers) + ".")
    return out


def chat_url() -> str:
    """Read at call time, not import time, so a changed MEDREASON_OLLAMA_URL takes effect."""
    return get_settings().ollama_url.rstrip("/") + "/api/chat"


def build_prompt(passage: dict, topic: str) -> tuple[str, str]:
    """The (system, user) text sent to the model for one passage. Kept separate so the exact prompt can be
    stored with the answer, which is what makes a written sentence traceable afterwards."""
    return SYSTEM, f"TOPIC: {topic}\n\nPASSAGE ({passage['title']}, section {passage['heading']}):\n{passage['text']}"


def _ask(passage: dict, topic: str, model: str, timeout: int) -> str:
    system, user = build_prompt(passage, topic)
    body = {"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(chat_url(), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"].strip()


def write_explanation(agent: dict, plan: dict, evidence: dict, model: str | None = None, timeout: int = 300,
                      recorder=None) -> dict:
    """`recorder`, if given, is called after every model call as recorder(model=, purpose=, prompt=,
    response=, seconds=). The storage layer passes one that writes the llm_calls table."""
    model = model or get_settings().ollama_model
    background, not_addressed = [], []
    for p in evidence["passages"]:
        topic = "; ".join(p["queries"])
        t0 = time.perf_counter()
        reply = _ask(p, topic, model, timeout)
        if recorder:
            system, user = build_prompt(p, topic)
            recorder(model=model, purpose="background_sentence", prompt=f"{system}\n\n{user}", response=reply,
                     seconds=time.perf_counter() - t0)
        sentence = re.sub(r"\s+", " ", reply).strip().strip('"')
        on_topic = not p.get("terms") or any(t in sentence.lower() for t in p["terms"])
        if "NOT_ADDRESSED" in sentence or not sentence or not on_topic:
            not_addressed.append(p["n"])
        else:
            background.append({"n": p["n"], "text": sentence})
    # a driver counts as covered only if a passage fetched for it produced a kept sentence
    kept = {n for n in (b["n"] for b in background)}
    covered = {d for p in evidence["passages"] if p["n"] in kept for d in p["drivers"]}
    missing = [q["driver"] for q in plan["queries"] if q["driver"] not in covered]
    return {"predicted_class": agent["belief"]["predicted_class"],   # copied from the agent, never from model text
            "case_sentences": case_sentences(agent, plan),
            "background": background,              # LLM-written, one per passage, still to be verified
            "not_addressed": not_addressed,        # passages the model said did not cover the topic
            "no_evidence_for": missing, "model": model}
