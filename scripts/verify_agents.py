"""Run: python scripts/verify_agents.py
Demos the Knowledge Manager (RAG) agent and Pedagogical Guide agent against
the already-ingested documents and the live Neo4j graph (run
scripts/verify_full_pipeline.py first so there's content to query). Builds
the content index fresh in-memory each run — fine at this scale; persisting
it is a follow-up, not attempted here (see MODULE_AGENTS_NOTES.md)."""

from __future__ import annotations

import spacy
import yake
from keybert import KeyBERT

from edukg.agents.knowledge_manager.bm25_index import BM25Index
from edukg.agents.knowledge_manager.content_indexer import ContentIndexService
from edukg.agents.knowledge_manager.graph_reranker import GraphDistanceReranker
from edukg.agents.knowledge_manager.knowledge_manager_agent import KnowledgeManagerAgent
from edukg.agents.pedagogical_guide.escalation_log import EscalationLogStore
from edukg.agents.pedagogical_guide.keyword_intent_strategy import KeywordIntentStrategy
from edukg.agents.pedagogical_guide.llm_intent_classifier import LLMIntentClassifier
from edukg.agents.pedagogical_guide.pedagogical_guide_agent import PedagogicalGuideAgent
from edukg.agents.shared.concept_tagger import ConceptTagger
from edukg.core.logging_setup import configure_logging, get_logger
from edukg.infrastructure.faiss.faiss_vector_index import FAISSVectorIndexRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.infrastructure.neo4j.neo4j_graph_repository import Neo4jGraphRepository
from edukg.infrastructure.openai.azure_openai_embedding_provider import AzureOpenAIEmbeddingProvider
from edukg.infrastructure.openai.azure_openai_llm_provider import AzureOpenAILLMProvider
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor
from edukg.modules.entity_linking.linking_service import EntityLinkingService

configure_logging()
logger = get_logger("verify_agents")

SAMPLE_QUESTIONS = [
    "What is the role of genome sequencing in discovering new antibiotics?",
    "I'm confused about why microorganisms would produce antibiotics in the first place.",
    "hi",
]


def main() -> None:
    document_repo = FilesystemDocumentRepository()
    entity_repo = FilesystemEntityRepository()
    embedding_provider = AzureOpenAIEmbeddingProvider()
    llm_provider = AzureOpenAILLMProvider()
    graph_repo = Neo4jGraphRepository()

    nlp = spacy.load("en_core_web_sm")
    mention_extractor = MentionExtractor(nlp, KeyBERT(), yake.KeywordExtractor(lan="en", n=3, top=10))
    linking_service = EntityLinkingService(entity_repo, FAISSVectorIndexRepository(dimension=embedding_provider.dimension()), embedding_provider)
    concept_tagger = ConceptTagger(mention_extractor, linking_service)

    print("Building content index (dense + lexical) from all ingested documents...")
    content_index_service = ContentIndexService(
        document_repo,
        entity_repo,
        embedding_provider,
        dense_index=FAISSVectorIndexRepository(dimension=embedding_provider.dimension()),
        lexical_index=BM25Index(),
    )
    content_index = content_index_service.build()
    print(f"Indexed {len(content_index.chunks_by_id)} chunks, {len(content_index.chunk_concepts)} chunk-concept mappings.\n")

    knowledge_manager = KnowledgeManagerAgent(
        content_index, embedding_provider, concept_tagger, GraphDistanceReranker(graph_repo)
    )
    pedagogical_guide = PedagogicalGuideAgent(
        KeywordIntentStrategy(), LLMIntentClassifier(llm_provider), concept_tagger, EscalationLogStore()
    )

    try:
        for question in SAMPLE_QUESTIONS:
            print(f"{'=' * 70}\nQ: {question}")

            turn = pedagogical_guide.classify(question)
            print(
                f"[Pedagogical Guide] intent={turn.intent.value} confidence={turn.confidence:.2f} "
                f"escalated={turn.escalated_to_llm} concepts={len(turn.detected_concepts)}"
            )

            results = knowledge_manager.retrieve(question, top_k=3)
            print(f"[Knowledge Manager] {len(results)} ranked chunk(s):")
            for rank, result in enumerate(results, start=1):
                preview = result.chunk.text[:120].replace("\n", " ")
                print(
                    f"  {rank}. score={result.final_score:.3f} (fused={result.fused_score:.3f}, "
                    f"graph={result.graph_score:.3f}) — {preview}..."
                )
            print()
    finally:
        graph_repo.close()

    print("Agent verification PASSED")


if __name__ == "__main__":
    main()
