"""
Splits the downloaded corpus (data/corpus/*.json) into retrievable passages.

Strategies (compared in app/rag/eval_chunking.py, which is how the default was chosen):

  sentence  Sections come from the paper's own XML headings, then each section is cut at
            sentence boundaries into passages of about `target_words` words, repeating
            trailing sentence(s) at the start of the next passage so a statement that
            straddles a boundary still appears whole in one of them. This is a
            hierarchical split (section, then sentence) like recursive character splitting,
            but it never cuts mid-sentence and sizes are in words, not characters.
  section   One passage per section, no splitting. The embedding model truncates what it
            cannot fit, so long sections are only partly represented.
  semantic  Sentences are embedded and a new passage starts where the similarity between
            neighbouring sentences drops below a percentile threshold (with min and max
            size). Needs an embedding function, so it costs more at build time.

Every passage keeps its paper's citation, which is what lets the final explanation say
where a claim came from. The embedded text is "<paper title>. <section heading>. <passage>"
rather than the passage alone: "the cut-off was 3" is meaningless without knowing it is
about the neutrophil-lymphocyte ratio in lung cancer. The display text stays the passage.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
TARGET_WORDS = 300      # about 400 tokens, chosen in app/rag/eval_chunking.py
OVERLAP_WORDS = 45      # about 15 percent
MIN_WORDS = 40

# split after . ! ? when the next token starts a sentence; keeps abbreviations like "vs." mostly intact
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Chunk:
    chunk_id: int
    pmcid: str
    topic: str
    heading: str
    text: str            # what is shown and quoted
    embed_text: str      # what is embedded (title + heading + text)
    citation: str        # short form for inline use
    title: str
    url: str
    license: str


def _citation(doc: dict) -> str:
    authors = doc.get("authors", "").strip().rstrip(".")
    first = authors.split(",")[0].strip() if authors else "Unknown"
    et_al = " et al." if "," in authors else ""
    journal = (doc.get("journal") or "").strip()
    return f"{first}{et_al}, {journal} {doc.get('year', '')}".strip().rstrip(",") + f" ({doc['pmcid']})"


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def _pack(sentences: list[str], target_words: int, overlap_words: int | None) -> list[str]:
    """Greedy sentence packing. overlap_words=None repeats exactly one sentence; an int
    repeats trailing sentences until at least that many words are carried over."""
    passages: list[str] = []
    cur: list[str] = []
    words = 0
    carried = 0          # how many leading sentences of `cur` are overlap from the previous passage
    for s in sentences:
        n = len(s.split())
        if cur and words + n > target_words:
            passages.append(" ".join(cur))
            keep: list[str] = []
            kw = 0
            for prev in reversed(cur):
                keep.insert(0, prev)
                kw += len(prev.split())
                if overlap_words is None or kw >= overlap_words:
                    break
            if len(keep) == len(cur) and len(cur) > 1:
                keep = keep[-1:]
                kw = len(keep[0].split())
            cur, words, carried = keep, kw, len(keep)
        cur.append(s)
        words += n
    if cur and len(cur) > carried:
        fresh = cur[carried:]
        tail = " ".join(cur)
        # fold a tiny trailing remainder into the previous passage instead of leaving a stub
        if passages and len(" ".join(fresh).split()) < MIN_WORDS:
            passages[-1] += " " + " ".join(fresh)
        else:
            passages.append(tail)
    return passages


def _semantic(sentences: list[str], embed: Callable[[list[str]], np.ndarray], threshold: float,
              min_words: int, max_words: int) -> list[str]:
    if len(sentences) < 2:
        return [" ".join(sentences)] if sentences else []
    vecs = embed(sentences)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)          # cosine, vectors are normalised
    passages, cur, words = [], [sentences[0]], len(sentences[0].split())
    for i in range(1, len(sentences)):
        n = len(sentences[i].split())
        if (sims[i - 1] < threshold and words >= min_words) or words + n > max_words:
            passages.append(" ".join(cur))
            cur, words = [], 0
        cur.append(sentences[i])
        words += n
    if cur:
        if passages and words < MIN_WORDS:
            passages[-1] += " " + " ".join(cur)
        else:
            passages.append(" ".join(cur))
    return passages


def load_chunks(corpus_dir: Path = CORPUS_DIR, strategy: str = "sentence",
                target_words: int = TARGET_WORDS, overlap_words: int | None = OVERLAP_WORDS,
                embed: Callable[[list[str]], np.ndarray] | None = None,
                semantic_percentile: float = 25.0, semantic_max_words: int = 300) -> list[Chunk]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(corpus_dir.glob("PMC*.json"))]

    threshold = 0.0
    if strategy == "semantic":
        assert embed is not None, "semantic chunking needs an embedding function"
        all_sims = []
        for d in docs:
            for sec in d["sections"]:
                s = _sentences(sec["text"])
                if len(s) > 1:
                    v = embed(s)
                    all_sims.extend(np.sum(v[:-1] * v[1:], axis=1).tolist())
        threshold = float(np.percentile(all_sims, semantic_percentile))

    chunks: list[Chunk] = []
    for doc in docs:
        cite = _citation(doc)
        for sec in doc["sections"]:
            if strategy == "section":
                passages = [sec["text"]]
            elif strategy == "semantic":
                passages = _semantic(_sentences(sec["text"]), embed, threshold, MIN_WORDS + 20, semantic_max_words)
            else:
                passages = _pack(_sentences(sec["text"]), target_words, overlap_words)
            for passage in passages:
                if len(passage.split()) < MIN_WORDS:
                    continue
                chunks.append(Chunk(
                    chunk_id=len(chunks), pmcid=doc["pmcid"], topic=doc["topic"],
                    heading=sec["heading"], text=passage,
                    embed_text=f"{doc['title']}. {sec['heading']}. {passage}",
                    citation=cite, title=doc["title"], url=doc["url"], license=doc["license"],
                ))
    return chunks


if __name__ == "__main__":
    for kw in [dict(), dict(target_words=170, overlap_words=None), dict(target_words=100, overlap_words=None), dict(strategy="section")]:
        cs = load_chunks(**kw)
        lens = sorted(len(c.text.split()) for c in cs)
        print(f"{str(kw):<55} {len(cs):>4} chunks, words min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
