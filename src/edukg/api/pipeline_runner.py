from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from edukg.api.state import AppResources
from edukg.core.config import get_settings
from edukg.core.logging_setup import get_logger
from edukg.core.models import Document
from edukg.core.models.enums import DocumentType, PipelineStage, Subject
from edukg.modules.entity_extraction.stage_handler import EntityExtractionStageHandler
from edukg.modules.entity_linking.linking_service import EntityLinkingService
from edukg.modules.entity_linking.stage_handler import EntityLinkingStageHandler
from edukg.modules.fusion.fusion_service import KnowledgeFusionService
from edukg.modules.fusion.stage_handler import KnowledgeFusionStageHandler
from edukg.modules.ingestion.ingestion_service import DocumentIngestionService
from edukg.modules.ontology.ontology_extension_service import OntologyExtensionService
from edukg.modules.ontology.ontology_merge_service import OntologyMergeService
from edukg.modules.ontology.ontology_review_service import OntologyReviewService
from edukg.modules.ontology.ontology_validator import OntologyValidationService
from edukg.modules.ontology.stage_handlers import (
    OntologyExtensionStageHandler,
    OntologyLoadStageHandler,
    OntologyValidationStageHandler,
)
from edukg.modules.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from edukg.modules.relation_extraction.relation_extraction_service import RelationExtractionService
from edukg.modules.relation_extraction.stage_handler import RelationExtractionStageHandler
from edukg.modules.storage.graph_storage_service import GraphStorageService
from edukg.modules.storage.stage_handler import GraphStorageStageHandler
from edukg.modules.subject_detection.keyword_strategy import KeywordSubjectDetectionStrategy
from edukg.modules.subject_detection.subject_detection_service import SubjectDetectionStageHandler
from edukg.modules.triple_generation.stage_handler import TripleGenerationStageHandler
from edukg.modules.triple_generation.triple_generator import TripleGenerator
from edukg.modules.triple_validation.domain_range_validator import TripleValidationService
from edukg.modules.triple_validation.stage_handler import TripleValidationStageHandler
from edukg.pipeline import CheckpointStore, PipelineContext, PipelineOrchestrator

logger = get_logger(__name__)

_EXTENSION_TO_TYPE = {".pdf": DocumentType.PDF, ".docx": DocumentType.DOCX}


@dataclass(slots=True)
class StageEvent:
    stage: str
    status: str
    timestamp: datetime


@dataclass(slots=True)
class PipelineRun:
    run_id: UUID
    state: str = "running"  # running | completed | failed
    events: list[StageEvent] = field(default_factory=list)
    error: str | None = None
    summary: dict | None = None
    event_queue: queue.Queue = field(default_factory=queue.Queue)


def _materialize_transitive_relations(resources: AppResources, subject: Subject) -> None:
    """Stage 12's "transitive reasoning" (Cypher-based multi-hop prerequisite
    chaining, per the pipeline spec) — runs once per transitive relation in
    the active ontology after Graph Storage completes, so chains like
    A-is_prerequisite_of->B-is_prerequisite_of->C automatically also get an
    inferred A->C edge, kept current as documents are added rather than a
    one-off manual script. force_reload=True: a freshly-approved ontology
    extension (or this run's own) must not read a stale pre-run cache."""
    active_ontology = resources.ontology_manager.get_active_ontology(subject, force_reload=True)
    for relation in active_ontology.relations:
        if relation.is_transitive:
            count = resources.graph_repo.materialize_transitive_closure(relation.name)
            logger.info("transitive_relation_materialized", predicate=relation.name, inferred_edge_count=count)


class PipelineRunRegistry:
    """In-memory registry of pipeline runs started via the API, keyed by
    run_id. Single-process/in-memory is a deliberate scope choice for a
    review-demo dashboard — nothing here needs to survive a process restart;
    a real deployment would move this to Redis/a DB."""

    def __init__(self) -> None:
        self._runs: dict[UUID, PipelineRun] = {}
        self._lock = threading.Lock()

    def create(self) -> PipelineRun:
        run = PipelineRun(run_id=uuid4())
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: UUID) -> PipelineRun | None:
        with self._lock:
            return self._runs.get(run_id)


def start_pipeline_run(resources: AppResources, registry: PipelineRunRegistry, relative_path: str) -> PipelineRun:
    run = registry.create()
    thread = threading.Thread(target=_execute, args=(resources, run, relative_path), daemon=True)
    thread.start()
    return run


def _execute(resources: AppResources, run: PipelineRun, relative_path: str) -> None:
    file_path = get_settings().paths.raw_dir / relative_path

    def on_stage_event(stage: PipelineStage, status: str) -> None:
        event = StageEvent(stage=stage.value, status=status, timestamp=datetime.now(timezone.utc))
        run.events.append(event)
        run.event_queue.put(event)

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        document_type = _EXTENSION_TO_TYPE.get(file_path.suffix.lower())
        if document_type is None:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Cheap per-run wiring — every heavy object (nlp, embedding/LLM
        # providers, repositories) is reused from AppResources, built once at
        # API startup. Entity linking reuses AppResources.entity_linking_index
        # — the same persistent, pre-populated index query-time tagging
        # (ConceptTagger) searches — rather than a fresh empty one per run;
        # a fresh one could never merge against entities from earlier runs
        # via similarity (only exact canonical-name matches would still
        # work), and newly created entities wouldn't be query-taggable until
        # a full process restart. See state.py's _populate_entity_linking_index.
        merge_service = OntologyMergeService(resources.ontology_manager)
        review_service = OntologyReviewService(resources.ontology_repo, merge_service)
        linking_service = EntityLinkingService(
            resources.entity_repo, resources.entity_linking_index, resources.embedding_provider
        )

        orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore())
        orchestrator.register(
            PipelineStage.INGESTION,
            DocumentIngestionService(resources.document_repo, resources.cache_repo, resources.ocr_provider).run,
        )
        orchestrator.register(
            PipelineStage.SUBJECT_DETECTION, SubjectDetectionStageHandler(KeywordSubjectDetectionStrategy()).run
        )
        orchestrator.register(PipelineStage.ONTOLOGY_LOAD, OntologyLoadStageHandler(resources.ontology_manager).run)
        orchestrator.register(
            PipelineStage.ONTOLOGY_EXTENSION,
            OntologyExtensionStageHandler(
                OntologyExtensionService(resources.llm_provider), resources.ontology_manager, resources.ontology_repo
            ).run,
        )
        orchestrator.register(
            PipelineStage.ONTOLOGY_VALIDATION,
            OntologyValidationStageHandler(OntologyValidationService(), review_service, resources.ontology_manager).run,
        )
        orchestrator.register(
            PipelineStage.ENTITY_EXTRACTION, EntityExtractionStageHandler(resources.mention_extractor).run
        )
        orchestrator.register(
            PipelineStage.ENTITY_LINKING, EntityLinkingStageHandler(linking_service, resources.ontology_manager).run
        )
        orchestrator.register(
            PipelineStage.RELATION_EXTRACTION,
            RelationExtractionStageHandler(
                RelationExtractionService(
                    resources.nlp, llm_relation_extractor=LLMRelationExtractor(resources.llm_provider)
                ),
                resources.ontology_manager,
            ).run,
        )
        orchestrator.register(PipelineStage.TRIPLE_GENERATION, TripleGenerationStageHandler(TripleGenerator()).run)
        orchestrator.register(
            PipelineStage.TRIPLE_VALIDATION,
            TripleValidationStageHandler(TripleValidationService(), resources.ontology_manager).run,
        )
        orchestrator.register(
            PipelineStage.KNOWLEDGE_FUSION,
            KnowledgeFusionStageHandler(KnowledgeFusionService(), resources.pending_triple_store).run,
        )
        orchestrator.register(
            PipelineStage.GRAPH_STORAGE, GraphStorageStageHandler(GraphStorageService(resources.graph_repo)).run
        )

        placeholder = Document(file_path=file_path, file_hash="0" * 64, document_type=document_type)
        context = orchestrator.run(PipelineContext(document=placeholder), on_stage_event=on_stage_event)

        run.summary = {
            "document_id": context.document.id,
            "chunk_count": len(context.chunks),
            "mention_count": len(context.mentions),
            "entity_count": len(context.entities),
            "relation_count": len(context.relations),
            "triple_count": len(context.triples),
        }

        try:
            resources.refresh_content_index()
        except Exception as exc:  # noqa: BLE001 - the ingestion itself succeeded; a failed index
            # refresh must not report the whole run as failed, only get logged. The Live Demo
            # simply won't see this document until the next successful refresh or restart.
            logger.error("content_index_refresh_failed", run_id=str(run.run_id), error=str(exc))

        try:
            _materialize_transitive_relations(resources, context.document.detected_subject)
        except Exception as exc:  # noqa: BLE001 - same reasoning as refresh_content_index above:
            # ingestion already succeeded, this is a post-processing enhancement, not grounds to
            # fail the whole run.
            logger.error("transitive_closure_failed", run_id=str(run.run_id), error=str(exc))

        run.state = "completed"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the dashboard, don't crash the thread silently
        logger.error("pipeline_run_failed", run_id=str(run.run_id), error=str(exc))
        run.state = "failed"
        run.error = str(exc)
    finally:
        run.event_queue.put(None)  # sentinel: tells the SSE stream this run is over
