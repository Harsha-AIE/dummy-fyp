from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_BYTES = 1024 * 1024


def compute_file_hash(path: Path) -> str:
    """SHA-256 of raw file bytes, streamed to bound memory use on large PDFs."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
