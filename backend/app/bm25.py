"""BM25 lexical search over the Chroma chunks, fused with vector search (RRF).

Why: embeddings miss exact tokens (clause numbers, "article L.1234", company names);
BM25 catches them. Reciprocal Rank Fusion merges both rankings without score calibration.
"""
import re
import unicodedata

from rank_bm25 import BM25Okapi

from .stores import get_collection

_index: dict = {"count": -1, "bm25": None, "ids": [], "docs": [], "metas": []}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # strip accents
    return re.findall(r"\w+", text)


def _ensure_index() -> None:
    col = get_collection()
    count = col.count()
    if count == _index["count"]:
        return  # rebuilt only when ingestion changed the collection size
    data = col.get(include=["documents", "metadatas"])
    _index.update(
        count=count,
        ids=data["ids"],
        docs=data["documents"],
        metas=data["metadatas"],
        bm25=BM25Okapi([tokenize(d) for d in data["documents"]]) if data["documents"] else None,
    )


def bm25_search(query: str, k: int = 5) -> list[dict]:
    _ensure_index()
    if _index["bm25"] is None:
        return []
    scores = _index["bm25"].get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"id": _index["ids"][i], "text": _index["docs"][i], "source": _index["metas"][i].get("source"),
         "page": _index["metas"][i].get("page")}
        for i in top
        if scores[i] > 0
    ]


def vector_hits(query: str, k: int) -> list[dict]:
    res = get_collection().query(query_texts=[query], n_results=k)
    return [
        {"id": i, "text": t, "source": m.get("source"), "page": m.get("page")}
        for i, t, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]


def fused_search(query: str, k: int = 5, pool: int = 20, rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion of vector and BM25 rankings."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for hits in (vector_hits(query, pool), bm25_search(query, pool)):
        for rank, h in enumerate(hits):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            by_id[h["id"]] = h
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [by_id[i] for i in best]
