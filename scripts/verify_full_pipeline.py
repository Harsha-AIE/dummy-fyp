"""Run: python scripts/verify_full_pipeline.py
Runs the real PipelineOrchestrator through all 12 stages (Modules 1-5) against
data/raw/amazon.pdf, using real providers throughout: Azure OpenAI (chat +
embeddings), PaddleOCR, FAISS, and Neo4j. Requires .env populated with
AZURE_OPENAI_* and EDUKG_NEO4J_* values (see .env.example)."""

from __future__ import annotations

from pathlib import Path

import spacy
import yake
from keybert import KeyBERT

from edukg.core.logging_setup import configure_logging, get_logger
from edukg.core.models import Document
from edukg.core.models.enums import DocumentType, PipelineStage
from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository
from edukg.infrastructure.faiss.faiss_vector_index import FAISSVectorIndexRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository
from edukg.infrastructure.ocr.paddleocr_provider import PaddleOCRProvider
from edukg.infrastructure.openai.azure_openai_embedding_provider import AzureOpenAIEmbeddingProvider
from edukg.infrastructure.openai.azure_openai_llm_provider import AzureOpenAILLMProvider
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor
from edukg.modules.entity_extraction.stage_handler import EntityExtractionStageHandler
from edukg.modules.entity_linking.linking_service import EntityLinkingService
from edukg.modules.entity_linking.stage_handler import EntityLinkingStageHandler
from edukg.modules.fusion.fusion_service import KnowledgeFusionService
from edukg.modules.fusion.stage_handler import KnowledgeFusionStageHandler
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

configure_logging()
logger = get_logger("verify_full_pipeline")


def main() -> None:
    # NOTE: data/raw/amazon.pdf is actually a Word OOXML file (zip signature,
    # word/document.xml inside) mislabeled with a .pdf extension, AND its
    # content turns out to be the same two synthetic sentences
    # verify_module2.py's _make_sample_docx() writes as a smoke-test fixture
    # — not real content. data/raw/DNA Task 3.docx is real coursework content
    # (32 paragraphs on genome sequencing / antibiotic discovery) and makes a
    # far more honest demo of the pipeline actually working.
    sample_path = Path("data/raw/DNA Task 3.docx")
    if not sample_path.exists():
        raise SystemExit(f"Demo document not found: {sample_path}")

    # ---- shared NLP models (loaded once, reused across stages) ----
    nlp = spacy.load("en_core_web_sm")
    keybert_model = KeyBERT()
    yake_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)

    # ---- repositories / providers ----
    document_repo = FilesystemDocumentRepository()
    cache_repo = FilesystemCacheRepository()
    ontology_repo = FilesystemOntologyRepository()
    entity_repo = FilesystemEntityRepository()
    ontology_manager = OntologyManager(ontology_repo, cache_repo)

    llm_provider = AzureOpenAILLMProvider()
    embedding_provider = AzureOpenAIEmbeddingProvider()
    vector_index = FAISSVectorIndexRepository(dimension=embedding_provider.dimension())
    graph_repo = Neo4jGraphRepository()

    # ---- stage 1-5 (Modules 1-2, already delivered) ----
    ingestion = DocumentIngestionService(document_repo, cache_repo, PaddleOCRProvider())
    subject_detection = SubjectDetectionStageHandler(KeywordSubjectDetectionStrategy())
    ontology_load = OntologyLoadStageHandler(ontology_manager)
    extension_service = OntologyExtensionService(llm_provider)
    ontology_extension = OntologyExtensionStageHandler(extension_service, ontology_manager, ontology_repo)
    validator = OntologyValidationService()
    merge_service = OntologyMergeService(ontology_manager)
    review_service = OntologyReviewService(ontology_repo, merge_service)
    ontology_validation = OntologyValidationStageHandler(validator, review_service, ontology_manager)

    # ---- stage 6-12 (Modules 3-5, this pass) ----
    mention_extractor = MentionExtractor(nlp, keybert_model, yake_extractor)
    entity_extraction = EntityExtractionStageHandler(mention_extractor)

    linking_service = EntityLinkingService(entity_repo, vector_index, embedding_provider)
    entity_linking = EntityLinkingStageHandler(linking_service, ontology_manager)

    relation_service = RelationExtractionService(nlp)
    relation_extraction = RelationExtractionStageHandler(relation_service, ontology_manager)

    triple_generation = TripleGenerationStageHandler(TripleGenerator())

    triple_validation_service = TripleValidationService()
    triple_validation = TripleValidationStageHandler(triple_validation_service, ontology_manager)

    fusion_service = KnowledgeFusionService()
    knowledge_fusion = KnowledgeFusionStageHandler(fusion_service)

    graph_storage_service = GraphStorageService(graph_repo)
    graph_storage = GraphStorageStageHandler(graph_storage_service)

    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore())
    orchestrator.register(PipelineStage.INGESTION, ingestion.run)
    orchestrator.register(PipelineStage.SUBJECT_DETECTION, subject_detection.run)
    orchestrator.register(PipelineStage.ONTOLOGY_LOAD, ontology_load.run)
    orchestrator.register(PipelineStage.ONTOLOGY_EXTENSION, ontology_extension.run)
    orchestrator.register(PipelineStage.ONTOLOGY_VALIDATION, ontology_validation.run)
    orchestrator.register(PipelineStage.ENTITY_EXTRACTION, entity_extraction.run)
    orchestrator.register(PipelineStage.ENTITY_LINKING, entity_linking.run)
    orchestrator.register(PipelineStage.RELATION_EXTRACTION, relation_extraction.run)
    orchestrator.register(PipelineStage.TRIPLE_GENERATION, triple_generation.run)
    orchestrator.register(PipelineStage.TRIPLE_VALIDATION, triple_validation.run)
    orchestrator.register(PipelineStage.KNOWLEDGE_FUSION, knowledge_fusion.run)
    orchestrator.register(PipelineStage.GRAPH_STORAGE, graph_storage.run)

    placeholder = Document(file_path=sample_path, file_hash="0" * 64, document_type=DocumentType.DOCX)
    try:
        context = orchestrator.run(PipelineContext(document=placeholder))
    finally:
        graph_repo.close()

    logger.info(
        "full_pipeline_verification_complete",
        subject=context.document.detected_subject.value,
        chunk_count=len(context.chunks),
        mention_count=len(context.mentions),
        entity_count=len(context.entities),
        relation_count=len(context.relations),
        triple_count=len(context.triples),
        completed_stages=[s.value for s in context.completed_stages],
    )
    print(
        f"Full pipeline PASSED — chunks={len(context.chunks)} mentions={len(context.mentions)} "
        f"entities={len(context.entities)} relations={len(context.relations)} triples={len(context.triples)}"
    )
    print("Open Neo4j Browser and run: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100")


if __name__ == "__main__":
    main()
