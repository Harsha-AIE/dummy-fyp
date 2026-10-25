from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

from edukg.core.exceptions import CacheError, ExternalServiceError
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class BM25Index:
    """
    Lexical retrieval side of the Knowledge Manager agent's hybrid search
    (`rank_bm25.BM25Okapi`). Persisted as the tokenized corpus + chunk-id
    list rather than a pickled BM25Okapi object — avoids both a pickle
    security surface and rank_bm25 internal-format version skew; rebuilding
    BM25Okapi from the tokenized corpus on load is cheap.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_ids: list[UUID] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunk_ids: list[UUID], texts: list[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ExternalServiceError("The 'rank_bm25' package is not installed") from exc

        self._chunk_ids = list(chunk_ids)
        self._tokenized_corpus = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        logger.info("bm25_index_built", chunk_count=len(self._chunk_ids))

    def search(self, query: str, top_k: int) -> list[tuple[UUID, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        # No positive-score filter: rank_bm25's classic (non-"+1"-smoothed)
        # idf can legitimately be exactly 0, or even negative, for a term
        # that appears in roughly half a small corpus — that does not mean
        # "no match," only that the raw score isn't independently meaningful.
        # This index only ever feeds Reciprocal Rank Fusion, which uses rank
        # position, not the score value, so returning the top-k by relative
        # score is correct regardless of sign.
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_positions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in ranked_positions]

    def save(self, path: Path) -> None:
        payload = {"chunk_ids": [str(c) for c in self._chunk_ids], "tokenized_corpus": self._tokenized_corpus}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving BM25 index to {path}") from exc
        logger.info("bm25_index_saved", path=str(path), chunk_count=len(self._chunk_ids))

    def load(self, path: Path) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ExternalServiceError("The 'rank_bm25' package is not installed") from exc

        if not path.exists():
            logger.warning("bm25_index_missing_on_load", path=str(path))
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._chunk_ids = [UUID(c) for c in payload["chunk_ids"]]
        self._tokenized_corpus = payload["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        logger.info("bm25_index_loaded", path=str(path), chunk_count=len(self._chunk_ids))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.lower())
