"""
Embeds the corpus passages and stores them in Qdrant, then answers similarity queries.

By default Qdrant runs in embedded (local) mode, as a folder under data/qdrant/, so there is
nothing to install or start. Setting MEDREASON_QDRANT_URL (see app/config.py) switches to a
Qdrant server, which is what docker compose uses; collections, payloads and queries are the
same API either way. A server starts empty, so run `python -m app.rag.index --build` once with
the same setting to fill it.

The embedding model is a parameter because it was chosen by measurement, not assumed:
app/rag/eval_retrieval.py scores candidate models on hand-written clinical questions.
One folder per model, so models can be compared without rebuilding each other's index.

Every search result carries the passage, its section heading, and its paper's citation,
since the explanation writer has to be able to say where each claim came from.
"""
from __future__ import annotations

import torch  # noqa: F401  (import before sklearn/shap-adjacent code on Windows, see CLAUDE.md section 4)

import atexit
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.rag.chunker import Chunk, load_chunks

COLLECTION = "evidence"
DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"
# Passages scoring below this are treated as "no supporting evidence". Chosen from
# app/rag/eval_retrieval.py: for this model and chunking, every answerable test query's best
# passage scored >= 0.409 and every off-corpus query's best passage scored <= 0.276, and 0.34
# is the midpoint. The sample is small (19 answerable, 8 off-corpus) and one answerable query
# was added after it failed, so treat the floor as provisional and re-measure if the corpus
# or the chunking changes.
MIN_SCORE = 0.34

# some retrieval models expect an instruction in front of the query (not the documents)
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

_models: dict[str, SentenceTransformer] = {}
_clients: dict[str, QdrantClient] = {}
atexit.register(lambda: [c.close() for c in _clients.values()])  # avoids a noisy __del__ at interpreter exit


def _slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def _model(name: str) -> SentenceTransformer:
    if name not in _models:
        _models[name] = SentenceTransformer(name, device="cpu")
    return _models[name]


def collection(name: str) -> str:
    """Embedded mode keeps one folder per model, so the collection name is the same in each. A server
    holds every model in one place, so there the model is part of the name and two models' vectors
    (which can have different sizes) never land in the same collection."""
    return f"{COLLECTION}_{_slug(name)}" if get_settings().qdrant_url else COLLECTION


def _client(name: str) -> QdrantClient:
    if name not in _clients:
        s = get_settings()
        if s.qdrant_url:
            _clients[name] = QdrantClient(url=s.qdrant_url)
        else:
            path = s.qdrant_path / _slug(name)
            path.mkdir(parents=True, exist_ok=True)
            _clients[name] = QdrantClient(path=str(path))
    return _clients[name]


def build_index(model_name: str = DEFAULT_MODEL, chunks: list[Chunk] | None = None) -> int:
    chunks = chunks if chunks is not None else load_chunks()
    model, client, coll = _model(model_name), _client(model_name), collection(model_name)
    vectors = model.encode([c.embed_text for c in chunks], batch_size=32, normalize_embeddings=True,
                           show_progress_bar=False)
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))
    client.upsert(coll, points=[
        PointStruct(id=c.chunk_id, vector=v.tolist(), payload={
            "chunk_id": c.chunk_id, "pmcid": c.pmcid, "topic": c.topic, "heading": c.heading, "text": c.text,
            "citation": c.citation, "title": c.title, "url": c.url, "license": c.license,
        }) for c, v in zip(chunks, vectors)
    ])
    return len(chunks)


def search(query: str, k: int = 5, model_name: str = DEFAULT_MODEL, topic: str | None = None) -> list[dict]:
    q = _model(model_name).encode(QUERY_PREFIX.get(model_name, "") + query, normalize_embeddings=True)
    flt = Filter(must=[FieldCondition(key="topic", match=MatchValue(value=topic))]) if topic else None
    hits = _client(model_name).query_points(collection(model_name), query=q.tolist(), limit=k, query_filter=flt).points
    return [{"score": round(float(h.score), 4), **h.payload} for h in hits]


def retrieve(query: str, k: int = 5, model_name: str = DEFAULT_MODEL, topic: str | None = None,
             min_score: float = MIN_SCORE) -> list[dict]:
    """search() minus passages too weakly related to cite. An empty list means the corpus
    has nothing relevant, and the explanation must say so instead of citing weak matches."""
    return [h for h in search(query, k, model_name, topic) if h["score"] >= min_score]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the index first")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("query", nargs="?")
    a = ap.parse_args()
    if a.build:
        print(f"indexed {build_index(a.model)} passages with {a.model}")
    if a.query:
        for h in search(a.query, a.k, a.model):
            print(f"\n[{h['score']}] {h['citation']} | {h['heading']}\n  {h['text'][:260]}...")
