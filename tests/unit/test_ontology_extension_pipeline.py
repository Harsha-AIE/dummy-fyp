import json
from pathlib import Path
from uuid import uuid4

import pytest

from edukg.core.exceptions import OntologyExtensionError
from edukg.core.interfaces import LLMProvider
from edukg.core.models import Chunk, Document, DocumentType, Subject
from edukg.core.models.enums import ExtractionMethod
from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
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
from edukg.pipeline.context import PipelineContext


class _StubLLMProvider(LLMProvider):
    def __init__(self, response_payload: dict) -> None:
        self._payload = response_payload

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        return json.dumps(self._payload)

    def model_name(self) -> str:
        return "stub-model"


def _make_document_and_chunks(subject_text: str) -> tuple[Document, list[Chunk]]:
    doc = Document(file_path=Path("data/raw/sample.pdf"), file_hash="e" * 64, document_type=DocumentType.PDF)
    chunk = Chunk(
        document_id=doc.id,
        sequence_index=0,
        text=subject_text,
        char_start=0,
        char_end=len(subject_text),
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=max(1, len(subject_text) // 4),
    )
    return doc, [chunk]


def _make_manager(tmp_path: Path) -> tuple[OntologyManager, FilesystemOntologyRepository]:
    from edukg.core.models import OntologyClass

    repo = FilesystemOntologyRepository(base_dir=tmp_path / "ontologies")
    concept = repo.add_class(OntologyClass(name="Concept", subject=Subject.UNKNOWN, is_core=True))
    cache = FilesystemCacheRepository(base_dir=tmp_path / "cache")
    manager = OntologyManager(repo, cache)
    return manager, repo


def test_extension_service_parses_valid_llm_response(tmp_path: Path) -> None:
    manager, repo = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    doc, chunks = _make_document_and_chunks("Gradient descent minimizes the loss function over the dataset.")

    llm = _StubLLMProvider(
        {
            "proposals": [
                {
                    "kind": "class",
                    "proposed_name": "Dataset",
                    "proposed_parent_class_name": "Concept",
                    "proposed_domain_class_name": None,
                    "proposed_range_class_name": None,
                    "proposed_aliases": ["training data"],
                    "proposed_description": "A collection of training examples",
                    "justification": "Referenced repeatedly in the content",
                    "confidence": 0.85,
                }
            ]
        }
    )
    service = OntologyExtensionService(llm)
    extensions = service.propose_extensions(doc, chunks, active)

    assert len(extensions) == 1
    assert extensions[0].proposed_name == "Dataset"
    assert extensions[0].proposed_parent_class_id == active.get_class("Concept").id
    assert extensions[0].generator_model == "stub-model"


def test_extension_service_skips_malformed_proposals_without_failing(tmp_path: Path) -> None:
    manager, repo = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    doc, chunks = _make_document_and_chunks("Some content.")

    llm = _StubLLMProvider({"proposals": [{"kind": "class"}]})  # missing required fields
    service = OntologyExtensionService(llm)
    extensions = service.propose_extensions(doc, chunks, active)

    assert extensions == []


def test_extension_service_rejects_non_json_response(tmp_path: Path) -> None:
    manager, repo = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    doc, chunks = _make_document_and_chunks("Some content.")

    class BadLLM(LLMProvider):
        def complete_json(self, *, system_prompt, user_prompt, schema_name):
            return "not json"

        def model_name(self):
            return "bad-model"

    service = OntologyExtensionService(BadLLM())
    with pytest.raises(OntologyExtensionError):
        service.propose_extensions(doc, chunks, active)


def test_extension_service_returns_empty_for_no_chunks(tmp_path: Path) -> None:
    manager, repo = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    doc, _ = _make_document_and_chunks("irrelevant")

    service = OntologyExtensionService(_StubLLMProvider({"proposals": []}))
    assert service.propose_extensions(doc, [], active) == []


def test_full_ontology_stage_pipeline_end_to_end(tmp_path: Path) -> None:
    """OntologyLoad -> OntologyExtension -> OntologyValidation, wired exactly as the
    orchestrator would run them, proving the three stage handlers compose correctly."""
    manager, repo = _make_manager(tmp_path)
    doc, chunks = _make_document_and_chunks("Gradient descent minimizes the loss function.")

    llm = _StubLLMProvider(
        {
            "proposals": [
                {
                    "kind": "class",
                    "proposed_name": "Dataset",
                    "proposed_parent_class_name": "Concept",
                    "proposed_domain_class_name": None,
                    "proposed_range_class_name": None,
                    "proposed_aliases": [],
                    "proposed_description": "training data collection",
                    "justification": "mentioned in the content",
                    "confidence": 0.9,
                }
            ]
        }
    )
    extension_service = OntologyExtensionService(llm)
    validator = OntologyValidationService()
    merge_service = OntologyMergeService(manager)
    review_service = OntologyReviewService(repo, merge_service)

    context = PipelineContext(document=doc)
    context.chunks = chunks
    context.document.detected_subject = Subject.MACHINE_LEARNING

    context = OntologyLoadStageHandler(manager).run(context)
    assert context.ontology_classes  # core Concept present

    context = OntologyExtensionStageHandler(extension_service, manager, repo).run(context)
    assert len(context.proposed_extensions) == 1

    context = OntologyValidationStageHandler(validator, review_service, manager).run(context)
    assert len(context.accepted_extensions) == 1
    assert any(c.name == "Dataset" for c in context.ontology_classes)
