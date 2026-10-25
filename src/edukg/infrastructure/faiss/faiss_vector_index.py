from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

import numpy as np

from edukg.core.exceptions import CacheError, ExternalServiceError
from edukg.core.interfaces import VectorIndexRepository
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

# faiss and torch (pulled in transitively by sentence-transformers/keybert)
# both bundle their own OpenMP runtime on Windows; loading both in the same
# process aborts the interpreter with "OMP: Error #15" unless this is set
# before faiss is imported. Setdefault so a caller's own setting is respected.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class FAISSVectorIndexRepository(VectorIndexRepository):
    """
    FAISS-backed cosine similarity search used by entity linking (stage 7).

    Vectors are L2-normalized before insertion/search so a plain IndexFlatIP
    (inner product) computes cosine similarity. FAISS only deals in row
    positions, so a parallel `_ids` list maps row position -> entity UUID;
    `save`/`load` persist both the FAISS index and this id-list sidecar
    together so they never drift out of sync.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = self._new_index()
        self._ids: list[UUID] = []

    def _new_index(self):
        try:
            import faiss
        except ImportError as exc:
            raise ExternalServiceError("The 'faiss-cpu' package is not installed") from exc
        return faiss.IndexFlatIP(self._dimension)

    @staticmethod
    def _normalize(vector: list[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def add(self, entity_id: UUID, embedding: list[float]) -> None:
        vector = self._normalize(embedding).reshape(1, -1)
        self._index.add(vector)
        self._ids.append(entity_id)

    def search(self, embedding: list[float], top_k: int) -> list[tuple[UUID, float]]:
        if self._index.ntotal == 0:
            return []
        vector = self._normalize(embedding).reshape(1, -1)
        scores, positions = self._index.search(vector, min(top_k, self._index.ntotal))
        results: list[tuple[UUID, float]] = []
        for position, score in zip(positions[0], scores[0], strict=True):
            if position == -1:
                continue
            results.append((self._ids[position], float(score)))
        return results

    def save(self, path: Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ExternalServiceError("The 'faiss-cpu' package is not installed") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self._index, str(path))
            sidecar = path.with_suffix(path.suffix + ".ids.json")
            sidecar.write_text(json.dumps([str(i) for i in self._ids]), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed saving FAISS index to {path}") from exc
        logger.info("faiss_index_saved", path=str(path), count=len(self._ids))

    def load(self, path: Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ExternalServiceError("The 'faiss-cpu' package is not installed") from exc
        sidecar = path.with_suffix(path.suffix + ".ids.json")
        if not path.exists() or not sidecar.exists():
            logger.warning("faiss_index_missing_on_load", path=str(path))
            return
        self._index = faiss.read_index(str(path))
        self._ids = [UUID(i) for i in json.loads(sidecar.read_text(encoding="utf-8"))]
        logger.info("faiss_index_loaded", path=str(path), count=len(self._ids))
