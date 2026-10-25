from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConfidenceScore(BaseModel):
    """A single, explainable confidence value with the method that produced it."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    method: str = Field(min_length=1, description="e.g. 'cosine_similarity', 'llm_judge', 'rule_based'")
    reason: str | None = Field(default=None)

    def meets_threshold(self, threshold: float) -> bool:
        return self.value >= threshold


class Provenance(BaseModel):
    """Full traceability record for any derived artifact."""

    model_config = ConfigDict(frozen=True)

    source_document_id: UUID
    page_number: int | None = Field(default=None, ge=0)
    chunk_id: UUID | None = None
    extractor_name: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    extracted_at: datetime = Field(default_factory=utcnow)
    raw_span_start: int | None = Field(default=None, ge=0)
    raw_span_end: int | None = Field(default=None, ge=0)


class Identifiable(BaseModel):
    """Base mixin providing a stable UUID identity."""

    id: UUID = Field(default_factory=uuid4)


class Timestamped(BaseModel):
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def touch(self) -> None:
        self.updated_at = utcnow()
