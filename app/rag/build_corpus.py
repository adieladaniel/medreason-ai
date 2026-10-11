"""
Builds the Phase 6 evidence corpus from open-access papers listed in corpus_sources.json.

For each PMC ID this downloads the article's full text as JATS XML from Europe PMC's
REST API and its citation record (title, authors, journal, year, DOI, license), then
writes one JSON file per paper to data/corpus/:

    {pmcid, title, authors, journal, year, doi, license, url, topic, sections:[{heading, text}]}

Why Europe PMC and not scraping the article pages: it serves the open-access subset as
structured XML (headings and paragraphs are already separated, no HTML to clean), it
exposes the license per article, and it is stable enough to script against.

The output is git-ignored with the rest of data/. Several of these licenses are
non-commercial or share-alike, so the repo carries only the manifest and this script and
anyone can rebuild the corpus. Run from the repo root:

    python -m app.rag.build_corpus
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = Path(__file__).resolve().parent / "corpus_sources.json"
OUT_DIR = ROOT / "data" / "corpus"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# sections that are scaffolding rather than evidence
SKIP_HEADINGS = {"references", "acknowledgements", "acknowledgments", "funding", "conflict of interest",
                 "conflicts of interest", "competing interests", "author contributions", "ethics statement",
                 "data availability", "abbreviations", "supplementary material", "supplementary materials",
                 "consent for publication", "informed consent", "disclosure", "disclosures", "ethical approval",
                 "declaration of interests", "declarations", "availability of data and materials"}
ALLOWED_LICENSES = {"cc by", "cc by-nc", "cc by-nc-sa", "cc0"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "medreason-ai-corpus-builder"})
    return urllib.request.urlopen(req, timeout=60).read()


def _metadata(pmcid: str) -> dict:
    q = urllib.parse.quote(f"PMCID:{pmcid}")
    d = json.loads(_get(f"{API}/search?query={q}&resultType=core&format=json"))
    r = d["resultList"]["result"][0]
    return {
        "title": r["title"].strip().rstrip("."),
        "authors": r.get("authorString", ""),
        "journal": r.get("journalInfo", {}).get("journal", {}).get("title") or r.get("journalTitle", ""),
        "year": r.get("pubYear"),
        "doi": r.get("doi"),
        "license": (r.get("license") or "").lower(),
    }


def _text(el: ET.Element) -> str:
    """All text under an element, citations and cross-references dropped."""
    parts = []

    def walk(node: ET.Element):
        if node.tag in {"xref", "table-wrap", "fig", "supplementary-material"}:
            if node.tail:
                parts.append(node.tail)
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if node.tail:
            parts.append(node.tail)

    if el.text:
        parts.append(el.text)
    for child in el:
        walk(child)
    text = " ".join("".join(parts).split())
    # removing <xref> citation markers leaves empty brackets and stray separators behind
    text = re.sub(r"\(\s*[,;\s]*\)|\[\s*[,;\s]*\]", "", text)
    text = re.sub(r"\s+([,.;:])", r"", text)
    return " ".join(text.split())


def _sections(root: ET.Element) -> list[dict]:
    out = []
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        txt = " ".join(_text(p) for p in abstract.iter("p") if _text(p))
        if txt:
            out.append({"heading": "Abstract", "text": txt})

    body = root.find(".//body")
    if body is None:
        return out

    def visit(sec: ET.Element, trail: list[str]):
        title_el = sec.find("title")
        title = _text(title_el) if title_el is not None else ""
        here = trail + ([title] if title else [])
        if title.lower().strip(" .:") in SKIP_HEADINGS:
            return
        paras = [_text(p) for p in sec.findall("p")]
        paras = [p for p in paras if len(p) > 40]
        if paras:
            out.append({"heading": " > ".join(here) or "Body", "text": " ".join(paras)})
        for sub in sec.findall("sec"):
            visit(sub, here)

    top_secs = body.findall("sec")
    if top_secs:
        for s in top_secs:
            visit(s, [])
    loose = [_text(p) for p in body.findall("p") if len(_text(p)) > 40]
    if loose:
        out.append({"heading": "Body", "text": " ".join(loose)})
    return out


def build() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = []
    for src in manifest["sources"]:
        pmcid = src["pmcid"]
        meta = _metadata(pmcid)
        if meta["license"] not in ALLOWED_LICENSES:
            print(f"SKIP {pmcid}: license '{meta['license']}' is not in the allowed set")
            continue
        xml = _get(f"{API}/{pmcid}/fullTextXML")
        sections = _sections(ET.fromstring(xml))
        words = sum(len(s["text"].split()) for s in sections)
        doc = {
            "pmcid": pmcid, **meta, "topic": src["topic"], "role": src["role"],
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            "sections": sections,
        }
        (OUT_DIR / f"{pmcid}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        built.append(doc)
        title = meta["title"].encode("ascii", "replace").decode()
        print(f"ok   {pmcid}  {meta['license']:<12} {len(sections):>2} sections {words:>6} words  {title[:60]}")
        time.sleep(0.4)  # be polite to the API
    return built


if __name__ == "__main__":
    docs = build()
    print(f"\n{len(docs)} papers, {sum(len(s['text'].split()) for d in docs for s in d['sections'])} words total "
          f"-> {OUT_DIR}")
