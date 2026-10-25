"""Run: python scripts/verify_module2.py
Proves Module 2 stage handlers compose correctly through the real PipelineOrchestrator,
using a stub LLM provider (no OpenAI key required) and a stub OCR provider."""

from __future__ import annotations

import json
from pathlib import Path

import docx

from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import configure_logging, get_logger
from edukg.core.models import Document
from edukg.core.models.enums import DocumentType, PipelineStage
from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
from edukg.modules.ingestion.ingestion_service import DocumentIngestionService
from edukg.modules.ontology.ontology_extension_service import OntologyExtensionService
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.ontology.ontology_merge_service import OntologyMergeService
from edukg.modules.ontology.ontology_review_service import OntologyReviewService
from edukg.modules.ontology.ontology_validator import OntologyValidationService
from edukg.modules.ontology.stage_handlers import (
    OntologyExtensionStageHandler,
    OntologyLoadStageHandler,
    OntologyValidationStageHandler,
)
from edukg.modules.subject_detection.keyword_strategy import KeywordSubjectDetectionStrategy
from edukg.modules.subject_detection.subject_detection_service import SubjectDetectionStageHandler
from edukg.pipeline import CheckpointStore, PipelineContext, PipelineOrchestrator

configure_logging()
logger = get_logger("verify_module2")


class _StubLLMProvider(LLMProvider):
    """Avoids requiring a real OpenAI key for this smoke test."""

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        return json.dumps(
            {
                "proposals": [
                    {
                        "kind": "class",
                        "proposed_name": "TrainingLoop",
                        "proposed_parent_class_name": "Concept",
                        "proposed_domain_class_name": None,
                        "proposed_range_class_name": None,
                        "proposed_aliases": ["training procedure"],
                        "proposed_description": "The iterative loop that updates model parameters.",
                        "justification": "Referenced explicitly in the document content.",
                        "confidence": 0.9,
                    }
                ]
            }
        )

    def model_name(self) -> str:
        return "stub-model"


class _StubOCRProvider:
    def extract_text(self, image_path: Path, *, layout_aware: bool = True) -> list[dict]:
        return []


def _make_sample_docx(path: Path) -> None:
    d = docx.Document()
    d.add_paragraph("Gradient descent minimizes the loss function during neural network training.")
    d.add_paragraph("Backpropagation computes gradients efficiently across layers.")
    d.save(str(path))


def main() -> None:
    workspace = Path("data/tmp_module2_demo")
    workspace.mkdir(parents=True, exist_ok=True)
    sample_path = Path("data/raw/DNA Task 3.docx")
    _make_sample_docx(sample_path)

    document_repo = FilesystemDocumentRepository()
    cache_repo = FilesystemCacheRepository()
    ontology_repo = FilesystemOntologyRepository()
    ontology_manager = OntologyManager(ontology_repo, cache_repo)

    ingestion = DocumentIngestionService(document_repo, cache_repo, _StubOCRProvider())
    subject_detection = SubjectDetectionStageHandler(KeywordSubjectDetectionStrategy())
    ontology_load = OntologyLoadStageHandler(ontology_manager)
    extension_service = OntologyExtensionService(_StubLLMProvider())
    ontology_extension = OntologyExtensionStageHandler(extension_service, ontology_manager, ontology_repo)
    validator = OntologyValidationService()
    merge_service = OntologyMergeService(ontology_manager)
    review_service = OntologyReviewService(ontology_repo, merge_service)
    ontology_validation = OntologyValidationStageHandler(validator, review_service, ontology_manager)

    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore())
    orchestrator.register(PipelineStage.INGESTION, ingestion.run)
    orchestrator.register(PipelineStage.SUBJECT_DETECTION, subject_detection.run)
    orchestrator.register(PipelineStage.ONTOLOGY_LOAD, ontology_load.run)
    orchestrator.register(PipelineStage.ONTOLOGY_EXTENSION, ontology_extension.run)
    orchestrator.register(PipelineStage.ONTOLOGY_VALIDATION, ontology_validation.run)

    placeholder = Document(file_path=sample_path, file_hash="0" * 64, document_type=DocumentType.DOCX)
    context = orchestrator.run(PipelineContext(document=placeholder), stop_after=PipelineStage.ONTOLOGY_VALIDATION)

    logger.info(
        "module2_verification_complete",
        subject=context.document.detected_subject.value,
        subject_confidence=context.document.subject_confidence,
        chunk_count=len(context.chunks),
        ontology_class_count=len(context.ontology_classes),
        proposed_extensions=len(context.proposed_extensions),
        accepted_extensions=len(context.accepted_extensions),
        completed_stages=[s.value for s in context.completed_stages],
    )
    assert context.document.detected_subject.value == "machine_learning"
    assert len(context.accepted_extensions) == 0
    assert any(c.name == "TrainingLoop" for c in context.ontology_classes)
    print("Module 2 end-to-end verification PASSED")


if __name__ == "__main__":
    main()
