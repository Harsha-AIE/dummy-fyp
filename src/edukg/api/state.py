from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import FastAPI, Request

from edukg.agents.knowledge_manager.bm25_index import BM25Index
from edukg.agents.knowledge_manager.content_indexer import ContentIndex, ContentIndexService
from edukg.agents.knowledge_manager.graph_reranker import GraphDistanceReranker
from edukg.agents.knowledge_manager.knowledge_manager_agent import KnowledgeManagerAgent
from edukg.agents.knowledge_tracing.dkt_service import DKTInferenceService
from edukg.agents.knowledge_tracing.knowledge_tracing_agent import KnowledgeTracingAgent
from edukg.agents.knowledge_tracing.topic_recommender import TopicRecommender
from edukg.agents.pedagogical_guide.escalation_log import EscalationLogStore
from edukg.agents.pedagogical_guide.keyword_intent_strategy import KeywordIntentStrategy
from edukg.agents.pedagogical_guide.llm_intent_classifier import LLMIntentClassifier
from edukg.agents.pedagogical_guide.pedagogical_guide_agent import PedagogicalGuideAgent
from edukg.agents.shared.concept_tagger import ConceptTagger
from edukg.agents.tutor_orchestrator.tutor_response_agent import TutorResponseAgent
from edukg.core.logging_setup import get_logger
from edukg.infrastructure.faiss.faiss_vector_index import FAISSVectorIndexRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository
from edukg.infrastructure.ocr.paddleocr_provider import PaddleOCRProvider
from edukg.infrastructure.openai.azure_openai_embedding_provider import AzureOpenAIEmbeddingProvider
from edukg.infrastructure.openai.azure_openai_llm_provider import AzureOpenAILLMProvider
from edukg.infrastructure.postgres.kt_store import KnowledgeTracingStore
from edukg.infrastructure.postgres.postgres_document_store import PostgresDocumentStore
from edukg.infrastructure.postgres.profile_store import ProfileStore
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor
from edukg.modules.entity_linking.linking_service import EntityLinkingService
from edukg.modules.fusion.pending_triple_store import PendingTripleStore
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.ontology.ontology_merge_service import OntologyMergeService
from edukg.modules.ontology.ontology_review_service import OntologyReviewService

if TYPE_CHECKING:
    # spaCy/KeyBERT/yake are only imported for real inside _build_resources()
    # (below) — they pull in torch/transformers transitively, which costs
    # real seconds on every import. Routers only need AppResources for a type
    # hint, and must not pay that cost just to be imported by a unit test
    # that never touches these fields.
    from spacy.language import Language

logger = get_logger(__name__)


def _populate_entity_linking_index(entity_repo: FilesystemEntityRepository, index: FAISSVectorIndexRepository) -> int:
    """Query-time concept tagging (ConceptTagger) and ingestion-time entity
    linking both need every *existing* entity's embedding in this index to
    do similarity-based matching — a fresh, empty FAISSVectorIndexRepository
    (the previous behavior, both here and in pipeline_runner.py's per-run
    linking service) means only an exact canonical-name string match ever
    resolves; any paraphrase or reworded query silently tags nothing. Found
    via a real gap: "genetic material" never matched the existing "DNA"
    entity despite a real ~0.5 cosine similarity, because the index it was
    searched against had zero entities loaded into it. Returns the count
    loaded, for startup logging."""
    count = 0
    for entity in entity_repo.list_all():
        if entity.embedding is not None:
            index.add(entity.id, entity.embedding)
            count += 1
    return count


@dataclass(slots=True)
class AppResources:
    """Everything expensive to build, constructed exactly once at process
    startup and reused across every request — building a fresh spaCy
    pipeline / re-embedding every chunk / opening a new Neo4j driver per
    request would make the API unusably slow. Routers read these off
    `request.app.state.resources`, never construct their own."""

    document_repo: FilesystemDocumentRepository
    entity_repo: FilesystemEntityRepository
    ontology_repo: FilesystemOntologyRepository
    cache_repo: FilesystemCacheRepository
    ontology_manager: OntologyManager
    ontology_review_service: OntologyReviewService
    graph_repo: Neo4jGraphRepository
    embedding_provider: AzureOpenAIEmbeddingProvider
    llm_provider: AzureOpenAILLMProvider
    pending_triple_store: PendingTripleStore
    postgres_store: PostgresDocumentStore
    kt_store: KnowledgeTracingStore
    profile_store: ProfileStore
    dkt_service: DKTInferenceService
    content_index: ContentIndex
    content_index_service: ContentIndexService
    concept_tagger: ConceptTagger
    entity_linking_index: FAISSVectorIndexRepository
    knowledge_manager_agent: KnowledgeManagerAgent
    pedagogical_guide_agent: PedagogicalGuideAgent
    knowledge_tracing_agent: KnowledgeTracingAgent
    topic_recommender: TopicRecommender
    tutor_response_agent: TutorResponseAgent
    nlp: Language
    mention_extractor: MentionExtractor
    ocr_provider: PaddleOCRProvider

    def refresh_content_index(self) -> None:
        """Rebuilds the dense+lexical retrieval index from whatever is now in
        storage and swaps in a new KnowledgeManagerAgent built on it. Called
        after a pipeline run completes (pipeline_runner.py) so a freshly
        ingested document is searchable immediately — without this, a
        document uploaded and run through the dashboard would silently not
        show up in the Live Agent Demo until the process restarted."""
        self.content_index = self.content_index_service.build()
        self.knowledge_manager_agent = KnowledgeManagerAgent(
            self.content_index,
            self.embedding_provider,
            self.concept_tagger,
            GraphDistanceReranker(self.graph_repo),
            gamma=0.15,
            mastery_provider=self._mastery_provider,
        )
        logger.info("content_index_refreshed", chunk_count=len(self.content_index.chunks_by_id))

    def _mastery_provider(self, student_id: str) -> dict[UUID, float]:
        return {e.concept_id: e.ensemble_mastery for e in self.knowledge_tracing_agent.get_mastery_vector(student_id)}


def _build_resources() -> AppResources:
    import spacy
    import yake
    from keybert import KeyBERT

    logger.info("api_resources_building")

    document_repo = FilesystemDocumentRepository()
    entity_repo = FilesystemEntityRepository()
    ontology_repo = FilesystemOntologyRepository()
    cache_repo = FilesystemCacheRepository()
    ontology_manager = OntologyManager(ontology_repo, cache_repo)
    ontology_review_service = OntologyReviewService(ontology_repo, OntologyMergeService(ontology_manager))

    graph_repo = Neo4jGraphRepository()
    embedding_provider = AzureOpenAIEmbeddingProvider()
    llm_provider = AzureOpenAILLMProvider()

    postgres_store = PostgresDocumentStore()
    try:
        postgres_store.ensure_schema()
    except Exception as exc:  # noqa: BLE001 - Postgres being briefly unreachable must not take down
        # the whole API (graph/agents/review don't depend on it) — only uploads would then fail
        # loudly until connectivity returns, which is the right place for that failure to surface.
        logger.error("postgres_schema_ensure_failed", error=str(exc))

    kt_store = KnowledgeTracingStore()
    try:
        kt_store.ensure_schema()
    except Exception as exc:  # noqa: BLE001 - same reasoning as postgres_store above: Knowledge
        # Tracing degrades (no mastery recorded) rather than taking the whole API down.
        logger.error("kt_schema_ensure_failed", error=str(exc))
    dkt_service = DKTInferenceService()  # logs internally whether a trained model was found

    profile_store = ProfileStore()
    try:
        profile_store.ensure_schema()
    except Exception as exc:  # noqa: BLE001 - same reasoning as postgres_store/kt_store above: the
        # Teacher/Student role picker degrades (role never persists server-side) rather than taking
        # the whole API down.
        logger.error("profile_schema_ensure_failed", error=str(exc))

    nlp = spacy.load("en_core_web_sm")
    ocr_provider = PaddleOCRProvider()
    mention_extractor = MentionExtractor(nlp, KeyBERT(), yake.KeywordExtractor(lan="en", n=3, top=10))

    # Shared across query-time tagging (below) AND every pipeline run's
    # ingestion-time entity linking (pipeline_runner.py uses this same
    # object, not a fresh one) — one persistent, ever-growing index of every
    # entity the KG has ever produced, not a per-run/per-process scratch one.
    entity_linking_index = FAISSVectorIndexRepository(dimension=embedding_provider.dimension())
    loaded_count = _populate_entity_linking_index(entity_repo, entity_linking_index)
    linking_service = EntityLinkingService(entity_repo, entity_linking_index, embedding_provider)
    concept_tagger = ConceptTagger(mention_extractor, linking_service)
    logger.info("entity_linking_index_populated", entity_count=loaded_count)

    content_index_service = ContentIndexService(
        document_repo,
        entity_repo,
        embedding_provider,
        dense_index=FAISSVectorIndexRepository(dimension=embedding_provider.dimension()),
        lexical_index=BM25Index(),
    )
    try:
        content_index = content_index_service.build()
    except Exception as exc:  # noqa: BLE001 - same reasoning as postgres_store/kt_store above: an
        # Azure OpenAI hiccup at startup (real, observed failure mode — an embedding request can
        # fail for reasons that have nothing to do with the rest of the app) must not take down
        # Neo4j/graph/review/pipeline-upload, which don't need embeddings at all. The Knowledge
        # Manager simply retrieves nothing until a pipeline run's refresh_content_index() succeeds.
        logger.error("content_index_build_failed", error=str(exc))
        content_index = ContentIndex(
            chunks_by_id={}, dense_index=FAISSVectorIndexRepository(dimension=embedding_provider.dimension()), lexical_index=BM25Index()
        )

    knowledge_tracing_agent = KnowledgeTracingAgent(kt_store, entity_repo, dkt_service)
    topic_recommender = TopicRecommender(knowledge_tracing_agent, graph_repo, entity_repo)

    def _mastery_provider(student_id: str) -> dict[UUID, float]:
        return {e.concept_id: e.ensemble_mastery for e in knowledge_tracing_agent.get_mastery_vector(student_id)}

    knowledge_manager_agent = KnowledgeManagerAgent(
        content_index,
        embedding_provider,
        concept_tagger,
        GraphDistanceReranker(graph_repo),
        gamma=0.15,
        mastery_provider=_mastery_provider,
    )
    pedagogical_guide_agent = PedagogicalGuideAgent(
        KeywordIntentStrategy(), LLMIntentClassifier(llm_provider), concept_tagger, EscalationLogStore()
    )
    tutor_response_agent = TutorResponseAgent(llm_provider)

    logger.info("api_resources_ready", chunk_count=len(content_index.chunks_by_id))
    return AppResources(
        document_repo=document_repo,
        entity_repo=entity_repo,
        ontology_repo=ontology_repo,
        cache_repo=cache_repo,
        ontology_manager=ontology_manager,
        ontology_review_service=ontology_review_service,
        graph_repo=graph_repo,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        pending_triple_store=PendingTripleStore(),
        postgres_store=postgres_store,
        kt_store=kt_store,
        profile_store=profile_store,
        dkt_service=dkt_service,
        content_index=content_index,
        content_index_service=content_index_service,
        concept_tagger=concept_tagger,
        entity_linking_index=entity_linking_index,
        knowledge_manager_agent=knowledge_manager_agent,
        pedagogical_guide_agent=pedagogical_guide_agent,
        knowledge_tracing_agent=knowledge_tracing_agent,
        topic_recommender=topic_recommender,
        tutor_response_agent=tutor_response_agent,
        nlp=nlp,
        mention_extractor=mention_extractor,
        ocr_provider=ocr_provider,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.resources = _build_resources()
    try:
        yield
    finally:
        app.state.resources.graph_repo.close()
        logger.info("api_resources_closed")


def get_resources(request: Request) -> AppResources:
    return request.app.state.resources


__all__ = ["AppResources", "lifespan", "get_resources"]
