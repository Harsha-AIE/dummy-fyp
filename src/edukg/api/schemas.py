from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConceptOut(BaseModel):
    id: UUID
    canonical_name: str


class MasteryEstimateOut(BaseModel):
    concept_id: UUID
    concept_name: str
    bkt_mastery: float
    dkt_mastery: float | None
    ensemble_mastery: float
    explanation: str
    updated_at: datetime


class MasteryVectorResponse(BaseModel):
    student_id: str
    estimates: list[MasteryEstimateOut]


class TopicRecommendationOut(BaseModel):
    concept_id: UUID
    concept_name: str
    mastery: float
    reason: str  # "foundational" | "ready"
    unblocks: list[str] = Field(default_factory=list)


class RecommendationsResponse(BaseModel):
    student_id: str
    recommendations: list[TopicRecommendationOut]


class KTEventOut(BaseModel):
    id: UUID
    concept_id: UUID
    source: str
    correct: bool
    weight: float
    occurred_at: datetime


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1)
    student_id: str | None = None


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    escalated_to_llm: bool
    detected_concepts: list[ConceptOut]
    updated_mastery: list[MasteryEstimateOut] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    student_id: str | None = None


class RetrievedChunkOut(BaseModel):
    chunk_id: UUID
    text: str
    page_number: int | None
    fused_score: float
    graph_score: float
    final_score: float


class RetrieveResponse(BaseModel):
    results: list[RetrievedChunkOut]


class AskRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    student_id: str | None = None


class TutorResponseOut(BaseModel):
    reply: str
    guardrail_passed: bool
    guardrail_note: str | None = None


class AskResponse(BaseModel):
    classification: ClassifyResponse
    retrieval: RetrieveResponse
    tutor_response: TutorResponseOut


class GraphNodeOut(BaseModel):
    id: str
    name: str
    type: str


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    predicate: str
    confidence: float
    evidence_count: int
    inferred: bool = False


class GraphResponse(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


class PendingExtensionOut(BaseModel):
    id: UUID
    kind: str
    subject: str
    proposed_name: str
    proposed_description: str | None
    justification: str
    confidence: float


class PendingTripleOut(BaseModel):
    id: UUID
    subject_entity_id: UUID
    subject_name: str
    predicate: str
    object_entity_id: UUID
    object_name: str
    confidence: float
    validation_status: str


class PaginatedTriplesResponse(BaseModel):
    items: list[PendingTripleOut]
    total: int
    page: int
    page_size: int


class ReviewActionResponse(BaseModel):
    id: UUID
    status: str  # "approved" | "rejected"


class BulkApproveRequest(BaseModel):
    min_confidence: float = Field(ge=0.0, le=1.0, description="Approve every pending triple at or above this confidence")


class BulkRejectRequest(BaseModel):
    max_confidence: float = Field(ge=0.0, le=1.0, description="Reject every pending triple at or below this confidence")


class BulkActionResponse(BaseModel):
    count: int


class DashboardSummaryResponse(BaseModel):
    entity_count: int
    accepted_triple_count: int
    pending_triple_count: int
    pending_extension_count: int


class PipelineRunRequest(BaseModel):
    path: str = Field(description="Path relative to data/raw, e.g. 'DNA Task 3.docx'")

    @field_validator("path")
    @classmethod
    def path_must_be_a_bare_filename(cls, v: str) -> str:
        # Mirrors the defense already applied to the upload endpoint
        # (routers/pipeline.py: Path(file.filename).name) — without this, a
        # client could pass "../../etc/passwd" or an absolute path and this
        # endpoint would hash/chunk/ingest any file on the host filesystem
        # that the process can read, since pipeline_runner.py joins this
        # value directly onto raw_dir with no further checks.
        if not v or Path(v).name != v:
            raise ValueError("path must be a bare filename inside data/raw, not a path with directory components")
        return v


class PipelineRunCreated(BaseModel):
    run_id: UUID


class PipelineStageEventOut(BaseModel):
    stage: str
    status: str
    timestamp: datetime


class PipelineRunSummary(BaseModel):
    document_id: UUID
    chunk_count: int
    mention_count: int
    entity_count: int
    relation_count: int
    triple_count: int


class PipelineRunStatus(BaseModel):
    run_id: UUID
    state: str  # "running" | "completed" | "failed"
    events: list[PipelineStageEventOut]
    error: str | None = None
    summary: PipelineRunSummary | None = None


class QuizQuestionRequest(BaseModel):
    concept_id: UUID
    student_id: str = Field(min_length=1)


class QuizQuestionOut(BaseModel):
    question_id: UUID
    concept_id: UUID
    concept_name: str
    question_text: str
    options: list[str]


class QuizSubmitRequest(BaseModel):
    question_id: UUID
    selected_index: int = Field(ge=0)
    student_id: str = Field(min_length=1)


class QuizSubmitResponse(BaseModel):
    correct: bool
    correct_index: int
    explanation: str
    updated_mastery: MasteryEstimateOut | None = None


class QuizDiagnosticRequest(BaseModel):
    document_id: UUID
    student_id: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class QuizDiagnosticResponse(BaseModel):
    document_id: UUID
    questions: list[QuizQuestionOut]


class AvailableDocumentOut(BaseModel):
    name: str
    already_ingested: bool
    document_id: UUID | None = None


class UploadedDocumentOut(BaseModel):
    name: str


class ProfileRegisterRequest(BaseModel):
    student_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1, max_length=80)
    # Literal, not a bare str — roster.py converts this straight to
    # ProfileRole(body.role); an unvalidated value would raise a bare
    # ValueError there (not an EduKGError), bypassing the app-wide error
    # handler and surfacing as an unhandled 500 instead of a clean 422.
    role: Literal["teacher", "student"]


class ProfileOut(BaseModel):
    student_id: str
    display_name: str
    role: str


class RosterEntryOut(BaseModel):
    student_id: str
    display_name: str
    concept_count: int
    average_mastery: float | None
    last_active: datetime | None


class RosterResponse(BaseModel):
    students: list[RosterEntryOut]
