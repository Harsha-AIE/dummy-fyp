from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class LLMProvider(ABC):
    """Abstraction over any chat-completion LLM backend (OpenAI today, swappable later)."""

    @abstractmethod
    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        """Return a raw JSON string conforming to the caller's expected schema.
        Implementations must enforce structured/JSON-mode output where supported."""
        raise NotImplementedError

    @abstractmethod
    def model_name(self) -> str: ...


class OCRProvider(ABC):
    @abstractmethod
    def extract_text(self, image_path: Path, *, layout_aware: bool) -> list[dict]:
        """Returns a list of layout blocks: {text, bbox, confidence, reading_order}."""
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def dimension(self) -> int: ...
