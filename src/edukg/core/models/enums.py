from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    TEXT = "text"


class ExtractionMethod(StrEnum):
    NATIVE_TEXT = "native_text"
    OCR_PADDLE = "ocr_paddle"
    HYBRID = "hybrid"


class Subject(StrEnum):
    MATHEMATICS = "mathematics"
    COMPUTER_SCIENCE = "computer_science"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGLISH = "english"
    UNKNOWN = "unknown"


class EntityType(StrEnum):
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    FORMULA = "formula"
    ALGORITHM = "algorithm"
    THEOREM = "theorem"
    DEFINITION = "definition"
    OTHER = "other"


class RelationDirection(StrEnum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class ValidationStatus(StrEnum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    MERGE = "merge"


class ReviewerRole(StrEnum):
    TEACHER = "teacher"
    ADMIN = "admin"
    STUDENT = "student"
    SYSTEM = "system"


class OntologyExtensionKind(StrEnum):
    CLASS = "class"
    RELATION = "relation"
    ALIAS = "alias"
    CONSTRAINT = "constraint"


class IntentCategory(StrEnum):
    QUESTION = "question"
    MISCONCEPTION = "misconception"
    EXAMPLE_REQUEST = "example_request"
    OFF_TOPIC = "off_topic"
    UNKNOWN = "unknown"


class KTEventSource(StrEnum):
    DIALOGUE = "dialogue"
    WORKSHEET = "worksheet"
    QUIZ = "quiz"


class ProfileRole(StrEnum):
    """A browser's self-selected account type (Teacher/Student role picker)
    — deliberately separate from ReviewerRole (who approved/rejected a
    triple in the review workflow, which carries ADMIN/SYSTEM members
    meaningless here) even though both have TEACHER/STUDENT members."""

    TEACHER = "teacher"
    STUDENT = "student"


class PipelineStage(StrEnum):
    INGESTION = "ingestion"
    SUBJECT_DETECTION = "subject_detection"
    ONTOLOGY_LOAD = "ontology_load"
    ONTOLOGY_EXTENSION = "ontology_extension"
    ONTOLOGY_VALIDATION = "ontology_validation"
    ENTITY_EXTRACTION = "entity_extraction"
    ENTITY_LINKING = "entity_linking"
    RELATION_EXTRACTION = "relation_extraction"
    TRIPLE_GENERATION = "triple_generation"
    TRIPLE_VALIDATION = "triple_validation"
    KNOWLEDGE_FUSION = "knowledge_fusion"
    GRAPH_STORAGE = "graph_storage"
