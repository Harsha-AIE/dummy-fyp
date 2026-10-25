from __future__ import annotations

import time
from pathlib import Path

from edukg.core.config import get_settings
from edukg.core.exceptions import CacheError
from edukg.core.interfaces import CacheRepository
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)


class FilesystemCacheRepository(CacheRepository):
    """Simple filesystem key-value cache. Keys are sanitized to safe filenames.
    A TTL, when given, is enforced lazily on read (expired entries are deleted)."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or get_settings().paths.cache_dir / "kv"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_key(self, key: str) -> str:
        return key.replace("/", "_").replace(":", "__").replace("\\", "_")

    def _value_path(self, key: str) -> Path:
        return self._base_dir / f"{self._safe_key(key)}.bin"

    def _meta_path(self, key: str) -> Path:
        return self._base_dir / f"{self._safe_key(key)}.meta"

    def get(self, key: str) -> bytes | None:
        value_path = self._value_path(key)
        meta_path = self._meta_path(key)
        if not value_path.exists():
            return None
        if meta_path.exists():
            expires_at = float(meta_path.read_text(encoding="utf-8").strip() or "0")
            if expires_at and time.time() > expires_at:
                value_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
        try:
            return value_path.read_bytes()
        except OSError as exc:
            raise CacheError(f"Failed reading cache key '{key}'") from exc

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        try:
            self._value_path(key).write_bytes(value)
            if ttl_seconds is not None:
                self._meta_path(key).write_text(str(time.time() + ttl_seconds), encoding="utf-8")
        except OSError as exc:
            raise CacheError(f"Failed writing cache key '{key}'") from exc
        logger.debug("cache_set", key=key, ttl_seconds=ttl_seconds)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def get_int(self, key: str, default: int = 0) -> int:
        raw = self.get(key)
        return int(raw.decode("utf-8")) if raw is not None else default

    def set_int(self, key: str, value: int) -> None:
        self.set(key, str(value).encode("utf-8"))

    def increment(self, key: str) -> int:
        new_value = self.get_int(key, default=0) + 1
        self.set_int(key, new_value)
        return new_value
