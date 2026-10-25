from __future__ import annotations

from uuid import UUID

_DEFAULT_K = 60


def reciprocal_rank_fusion(rankings: list[list[UUID]], k: int = _DEFAULT_K) -> list[tuple[UUID, float]]:
    """
    Combines multiple ranked lists of the same id type (here: chunk ids from
    dense vs. lexical retrieval) into one fused ranking, per the pipeline
    spec's "fuse: Reciprocal Rank Fusion" step. Each ranking contributes
    1/(k + rank) to an id's score (rank is 0-indexed here, so the top result
    of any ranking contributes 1/k) — an id near the top of *either* ranking
    scores well, without needing the two rankings' raw scores to be on a
    comparable scale (dense cosine similarity and BM25 scores are not).
    Returns ids sorted by fused score, descending.
    """
    scores: dict[UUID, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
