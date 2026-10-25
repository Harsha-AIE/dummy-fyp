from __future__ import annotations

from uuid import UUID

from edukg.core.config import ChunkingSettings, get_settings
from edukg.core.models import Chunk
from edukg.core.models.enums import ExtractionMethod

# Simple, dependency-free token approximation (~4 chars/token for English), consistent
# with how OpenAI-family tokenizers behave closely enough for chunk-sizing purposes.
_CHARS_PER_TOKEN = 4


class ChunkGenerator:
    """Splits cleaned page/document text into overlapping, bounded Chunks ready for
    downstream NLP. Chunking never crosses a page boundary — each page is chunked
    independently so `page_number` and provenance stay accurate."""

    def __init__(self, settings: ChunkingSettings | None = None) -> None:
        self._settings = settings or get_settings().chunking
        self._chunk_chars = self._settings.chunk_size_tokens * _CHARS_PER_TOKEN
        self._overlap_chars = self._settings.chunk_overlap_tokens * _CHARS_PER_TOKEN

    def generate(self, document_id: UUID, page_texts: list[tuple[int | None, str, ExtractionMethod]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        sequence_index = 0

        for page_number, text, extraction_method in page_texts:
            if not text.strip():
                continue
            for char_start, char_end in self._windows(len(text)):
                segment = text[char_start:char_end].strip()
                if not segment:
                    continue
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        sequence_index=sequence_index,
                        text=segment,
                        page_number=page_number,
                        char_start=char_start,
                        char_end=char_end,
                        extraction_method=extraction_method,
                        token_count=max(1, len(segment) // _CHARS_PER_TOKEN),
                    )
                )
                sequence_index += 1

        return chunks

    def _windows(self, text_length: int) -> list[tuple[int, int]]:
        if text_length <= self._chunk_chars:
            return [(0, text_length)]

        windows: list[tuple[int, int]] = []
        start = 0
        step = self._chunk_chars - self._overlap_chars
        while start < text_length:
            end = min(start + self._chunk_chars, text_length)
            windows.append((start, end))
            if end == text_length:
                break
            start += step
        return windows
