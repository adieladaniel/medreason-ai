"""
Compares claim checkers on the question that decides whether the explanation is useful: does the
checker keep the faithful sentences and catch the unfaithful ones?

Two labelled sets, each sentence judged against ONE passage:
  hand   the 35 hand-made sentences of eval_faithfulness.py (16 faithful, 19 unfaithful). Easy:
         the unfaithful ones were written to be wrong in a named way.
  real   the 41 sentences llama3.2 wrote in eval_writer.py (plain, no neighbour text). Labels were
         assigned by reading each sentence against the passage BEFORE any checker's verdict was
         looked at, and any claim whose support sat outside the printed lines was checked in the
         full passage. 36 faithful, 3 unfaithful (#27 two separate facts fused, #34 the group with
         the worst survival reversed, #35 a general survival figure attached to anemic patients),
         2 borderline (#17, #25) left out of the scores. Labelled by the project author with the
         assistant, one rater, so treat as a small sanity check, not a benchmark.

Checkers: "deberta-nli" is the current faithfulness.entailment (score = best-window entailment,
decision at 0.30); the MiniCheck variants are in minicheck.py (score = support probability,
decision at the paper's 0.5). No threshold is tuned here: each uses its own default, so the
comparison shows what each does out of the box. AUROC does not depend on a threshold.

Reported per checker: AUROC on each set, share of faithful sentences wrongly rejected, share of
unfaithful sentences caught, seconds per sentence (CPU). Scores are cached so a rerun is free.

Run:  python -m app.rag.eval_checker --backends deberta-nli,deberta-v3-large,flan-t5-large
"""
from __future__ import annotations

import torch  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent.parent.parent / "experiments" / "rag_retrieval"
CACHE = OUT / "checker_scores.json"
THRESHOLD = {"deberta-nli": 0.30}          # everything else: 0.5

# 1 faithful, 0 unfaithful, None borderline (excluded). Index = position in writer_items.json.
REAL_LABELS = {i: 1 for i in range(41)}
REAL_LABELS.update({27: 0, 34: 0, 35: 0, 17: None, 25: None})


def load_pairs() -> list[dict]:
    from app.rag.eval_faithfulness import SET
    hand_passages = json.loads((OUT / "faithfulness_passages.json").read_text(encoding="utf-8"))
    pairs = [{"id": f"hand:{i}", "set": "hand", "label": int(ok), "claim": s, "passage": hand_passages[k], "kind": kind}
             for i, (k, s, ok, kind) in enumerate(SET)]
    items = json.loads((OUT / "writer_items.json").read_text(encoding="utf-8"))
    rows = next(w for w in json.loads((OUT / "writer_eval.json").read_text(encoding="utf-8"))
                if w["model"] == "llama3.2:latest" and not w["expand"])["rows"]
    for i, (it, r) in enumerate(zip(items, rows)):
        if r["written"] and REAL_LABELS.get(i) is not None:
            pairs.append({"id": f"real:{i}", "set": "real", "label": REAL_LABELS[i], "claim": r["sentence"],
                          "passage": it["text"], "kind": "llama3.2 output"})
    return pairs


def scorer(name: str):
    if name == "deberta-nli":
        from app.rag import faithfulness
        return faithfulness.entailment
    from app.rag import minicheck
    return lambda claim, passage: minicheck.support_probability(name, passage, claim)


def score(name: str, pairs: list[dict], cache: dict) -> None:
    fn = scorer(name)
    cache.setdefault(name, {})
    for k, p in enumerate(pairs):
        if p["id"] in cache[name]:
            continue
        t = time.time()
        s = fn(p["claim"], p["passage"])
        cache[name][p["id"]] = {"score": s, "seconds": time.time() - t}
        if k % 10 == 0:
            print(f"  {name} {k + 1}/{len(pairs)}", flush=True)
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
    CACHE.write_text(json.dumps(cache), encoding="utf-8")


def summarise(names: list[str], pairs: list[dict], cache: dict) -> dict:
    out = {}
    for n in names:
        thr = THRESHOLD.get(n, 0.5)
        r = {"threshold": thr, "seconds": float(np.mean([cache[n][p["id"]]["seconds"] for p in pairs]))}
        for subset in ("hand", "real", "all"):
            ps = [p for p in pairs if subset == "all" or p["set"] == subset]
            y = np.array([p["label"] for p in ps])
            s = np.array([cache[n][p["id"]]["score"] for p in ps])
            ok = s >= thr
            r[subset] = {"n": len(ps), "faithful": int(y.sum()), "unfaithful": int((1 - y).sum()),
                         "auroc": float(roc_auc_score(y, s)),
                         "false_reject": float(1 - ok[y == 1].mean()),     # faithful sentences wrongly dropped
                         "caught": float((~ok[y == 0]).mean()),            # unfaithful sentences removed
                         "false_reject_n": int((~ok[y == 1]).sum()), "caught_n": int((~ok[y == 0]).sum())}
        out[n] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="deberta-nli")
    a = ap.parse_args()
    names = a.backends.split(",")
    pairs = load_pairs()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for n in names:
        print(f"scoring with {n}", flush=True)
        score(n, pairs, cache)
    res = summarise([n for n in cache if all(p["id"] in cache[n] for p in pairs)], pairs, cache)
    (OUT / "checker_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n{'checker':<18}{'thr':>5}  {'AUROC hand':>10} {'AUROC real':>10}  {'false-reject real':>18} {'false-reject hand':>18}  {'caught hand':>12} {'caught real':>12}  {'s/sent':>6}")
    for n, r in res.items():
        print(f"{n:<18}{r['threshold']:>5.2f}  {r['hand']['auroc']:>10.3f} {r['real']['auroc']:>10.3f}  "
              f"{r['real']['false_reject_n']:>3}/{r['real']['faithful']:<3} ({r['real']['false_reject']:.0%})".ljust(64) +
              f"{r['hand']['false_reject_n']:>3}/{r['hand']['faithful']:<3} ({r['hand']['false_reject']:.0%})   "
              f"{r['hand']['caught_n']:>3}/{r['hand']['unfaithful']:<3}  {r['real']['caught_n']:>3}/{r['real']['unfaithful']:<3}   {r['seconds']:>6.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
