"""Stage 2 of retrieval: a cross-encoder rescores the candidates from stage 1.

Stage 1 (bm25.fused_search) is cheap and recall-oriented: dense + BM25 fused with RRF, ~30 candidates.
Stage 2 reads each (question, chunk) pair jointly, which fixes what bi-encoders and BM25 miss
(negations such as "autorisé" vs "non autorisé", near-duplicate clauses). It is too slow to run on the
whole corpus, hence only on the shortlist.
"""
import logging

from . import config

log = logging.getLogger(__name__)

_model = None
_failed = False


def _get_model():
    global _model, _failed
    if _model is None and not _failed:
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(config.RERANK_MODEL, max_length=512)
        except Exception as e:  # missing dependency or download failure: degrade, don't crash
            _failed = True
            log.warning("Reranker unavailable (%s); falling back to stage-1 ranking.", e)
    return _model


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Return the k best candidates by cross-encoder score; stage-1 order if the model is unavailable."""
    if len(candidates) <= 1:
        return candidates[:k]
    model = _get_model()
    if model is None:
        return candidates[:k]
    scores = model.predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)[:k]
    return [{**c, "rerank_score": float(s)} for c, s in ranked]
