from __future__ import annotations


class EduKGError(Exception):
    """Base exception for all EduKG errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(EduKGError):
    """Raised when configuration is invalid or missing."""


class PipelineStageError(EduKGError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage_name: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(f"[{stage_name}] {message}", details=details)
        self.stage_name = stage_name


class DocumentProcessingError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("ingestion", message, details=details)


class SubjectDetectionError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("subject_detection", message, details=details)


class OntologyError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("ontology", message, details=details)


class OntologyExtensionError(OntologyError):
    pass


class OntologyValidationError(OntologyError):
    pass


class EntityExtractionError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("entity_extraction", message, details=details)


class EntityLinkingError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("entity_linking", message, details=details)


class RelationExtractionError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("relation_extraction", message, details=details)


class TripleGenerationError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("triple_generation", message, details=details)


class TripleValidationError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("triple_validation", message, details=details)


class KnowledgeFusionError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("knowledge_fusion", message, details=details)


class GraphStorageError(PipelineStageError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__("graph_storage", message, details=details)


class PedagogicalGuideError(EduKGError):
    """Raised when the Pedagogical Guide agent's LLM tier returns an
    unusable response (not a pipeline stage, so not a PipelineStageError)."""


class KnowledgeTracingError(EduKGError):
    """Raised when the Knowledge Tracing agent (BKT/DKT) fails to record an
    event or compute a mastery estimate (not a pipeline stage, so not a
    PipelineStageError — same reasoning as PedagogicalGuideError)."""


class TutorOrchestratorError(EduKGError):
    """Raised when the Tutor Response Orchestrator's LLM call fails or
    returns an unusable (non-JSON / missing-field) response (not a pipeline
    stage, so not a PipelineStageError — same reasoning as
    PedagogicalGuideError)."""


class QuizGenerationError(EduKGError):
    """Raised when the Quiz Agent's LLM call fails or returns an unusable
    (non-JSON / missing-field / wrong option count) question (not a
    pipeline stage, so not a PipelineStageError — same reasoning as
    PedagogicalGuideError)."""


class ExternalServiceError(EduKGError):
    """Raised when an external dependency (OpenAI, Neo4j, etc.) fails."""


class CacheError(EduKGError):
    pass
