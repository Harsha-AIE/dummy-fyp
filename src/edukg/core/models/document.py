from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field

from edukg.core.models.common import Identifiable, Timestamped
from edukg.core.models.enums import DocumentType, ExtractionMethod, Subject


class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    course: str | None = None
    language: str = Field(default="en")
    page_count: int | None = Field(default=None, ge=0)
    extra: dict[str, str] = Field(default_factory=dict)


class Document(Identifiable, Timestamped):
    """A single uploaded source document, tracked end-to-end through the pipeline."""

    file_path: Path
    file_hash: str = Field(min_length=1, description="SHA-256 hash of raw file bytes, used for caching")
    document_type: DocumentType
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    detected_subject: Subject = Field(default=Subject.UNKNOWN)
    subject_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Chunk(Identifiable):
    """A cleaned, bounded unit of text produced from a Document, ready for downstream NLP."""

    document_id: UUID
    sequence_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    extraction_method: ExtractionMethod
    token_count: int = Field(ge=0)

    def as_span(self) -> tuple[int, int]:
        return self.char_start, self.char_end


class Mention(Identifiable):
    """A raw textual occurrence of a candidate entity within a Chunk, prior to linking."""

    chunk_id: UUID
    document_id: UUID
    surface_form: str = Field(min_length=1)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)
    candidate_type: str | None = None
