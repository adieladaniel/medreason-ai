"""
Retrieval upgrades on top of the dense index in index.py, each switchable so they can be
measured one at a time (app/rag/eval_pipeline.py):

  expand   Appends known synonyms and spelled-out forms of clinical abbreviations to the
           query ("WBC" also gets "white blood cell count, leukocytosis"). Papers write
           "lactate dehydrogenase" or "LDH" unpredictably, and a planner that emits "elevated
           WBC" should still reach a paper that says "leukocytosis". The dictionary is
           written from clinical knowledge of the terms the models' top features map to,
           not tuned on the evaluation queries.
  hybrid   Dense search plus BM25 keyword search, merged by reciprocal rank fusion. Dense
           models capture meaning but blur exact terms ("LDH", "CURB-65"); BM25 is the
           opposite. RRF merges ranks, not scores, so the two scales never need aligning.
  rerank   A cross-encoder rereads the query together with each of the top candidates and
           rescores them. Slower than comparing vectors, so it only runs on the shortlist.

The dense cosine score stays available on every passage, because the "no evidence" floor in
index.py is defined on it. Reranker scores are exposed too, since they may abstain better.

All passages come from the Qdrant payloads, not a fresh chunking run, so BM25 and the dense
index are guaranteed to describe the same passages.
"""
from __future__ import annotations

import torch  # noqa: F401

import math
import re
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer

from app.rag import index as dense_index

_stem = PorterStemmer().stem
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = set("a an and are as at be been by for from has have in is it its of on or that the this to was were with which "
            "than then these those their they not can may also but if into more most such between during after before "
            "who whom whose how what when where why does do did our we i you your".split())

# Each key appears in clinical queries; values are alternate surface forms papers use.
EXPANSIONS: dict[str, list[str]] = {
    "wbc": ["white blood cell count", "leukocytosis"],
    "leukocytosis": ["white blood cell count", "WBC"],
    "leukocyte": ["white blood cell", "WBC"],
    "ldh": ["lactate dehydrogenase"],
    "nlr": ["neutrophil-to-lymphocyte ratio"],
    "plr": ["platelet-to-lymphocyte ratio"],
    "crp": ["C-reactive protein"],
    "pct": ["procalcitonin"],
    "bun": ["blood urea nitrogen", "urea"],
    "urea": ["blood urea nitrogen", "BUN"],
    "cxr": ["chest radiograph", "chest X-ray"],
    "cap": ["community-acquired pneumonia"],
    "nsclc": ["non-small cell lung cancer"],
    "sclc": ["small cell lung cancer"],
    "hb": ["hemoglobin", "anemia"],
    "hemoglobin": ["anemia"],
    "albumin": ["hypoalbuminemia"],
    "platelets": ["platelet count", "thrombocytosis"],
    "platelet": ["thrombocytosis"],
    "lymphocytes": ["lymphocyte count", "lymphopenia"],
    "lymphocyte": ["lymphopenia"],
    "sodium": ["hyponatremia"],
    "neutrophils": ["neutrophil count", "neutrophilia"],
    "mass": ["lung mass", "tumor"],
    "consolidation": ["airspace opacity", "infiltrate"],
    "edema": ["pulmonary edema"],
    "effusion": ["pleural effusion"],
    "atelectasis": ["lung collapse", "volume loss"],
    "cardiomegaly": ["enlarged heart", "heart failure"],
    "opacity": ["airspace opacity", "infiltrate"],
}


def expand_query(query: str) -> str:
    low = query.lower()
    extra: list[str] = []
    for term, alts in EXPANSIONS.items():
        if re.search(rf"\b{re.escape(term)}\b", low):
            extra += [a for a in alts if a.lower() not in low and a not in extra]
    return query if not extra else f"{query}. Also: {', '.join(extra)}"


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r + 1)
    return fused


_rerankers: dict = {}


def _reranker(name: str):
    if name not in _rerankers:
        from sentence_transformers import CrossEncoder
        _rerankers[name] = CrossEncoder(name, device="cpu", max_length=512)
    return _rerankers[name]


class Retriever:
    def __init__(self, model_name: str = dense_index.DEFAULT_MODEL):
        self.model_name = model_name
        client = dense_index._client(model_name)
        pts, offset = [], None
        while True:
            batch, offset = client.scroll(dense_index.collection(model_name), limit=256, offset=offset, with_payload=True)
            pts += batch
            if offset is None:
                break
        self.passages = sorted((p.payload for p in pts), key=lambda p: p["chunk_id"])
        assert [p["chunk_id"] for p in self.passages] == list(range(len(self.passages))), "chunk ids must be contiguous"
        self.bm25 = BM25([tokenize(f"{p['title']} {p['heading']} {p['text']}") for p in self.passages])

    def _dense_all(self, query: str) -> np.ndarray:
        hits = dense_index.search(query, k=len(self.passages), model_name=self.model_name)
        out = np.zeros(len(self.passages))
        for h in hits:
            out[h["chunk_id"]] = h["score"]
        return out

    def search(self, query: str, k: int = 5, expand: bool = False, mode: str = "dense",
               rerank: str | None = None, rerank_depth: int = 20) -> list[dict]:
        """mode: "dense", "lexical" (BM25 only), or "hybrid" (dense and BM25 merged by RRF)."""
        q = expand_query(query) if expand else query
        dense = self._dense_all(q)
        dense_rank = list(np.argsort(-dense))
        if mode in ("lexical", "hybrid"):
            lex = self.bm25.scores(tokenize(q))
            lex_rank = [i for i in np.argsort(-lex) if lex[i] > 0] or dense_rank
        if mode == "hybrid":
            fused = rrf([dense_rank, lex_rank])
            order = sorted(fused, key=lambda i: -fused[i])
        elif mode == "lexical":
            order = lex_rank + [i for i in dense_rank if i not in set(lex_rank)]
        else:
            order = dense_rank
        results = [{**self.passages[i], "dense_score": float(dense[i]), "rerank_score": None} for i in order[:max(k, rerank_depth if rerank else k)]]
        if rerank:
            ce = _reranker(rerank)
            # raw logits, not sigmoid probabilities: the sigmoid saturates near 1.0 for confident
            # matches, which makes a "no evidence" threshold hair-thin
            scores = ce.predict([(query, r["text"]) for r in results], show_progress_bar=False,
                                activation_fn=torch.nn.Identity())
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda r: -r["rerank_score"])
        return results[:k]
