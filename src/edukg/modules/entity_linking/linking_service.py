from __future__ import annotations

from uuid import UUID

from edukg.core.config import get_settings
from edukg.core.interfaces import EmbeddingProvider, EntityRepository, VectorIndexRepository
from edukg.core.logging_setup import get_logger
from edukg.core.models import Entity, Mention, Provenance
from edukg.core.models.enums import EntityType
from edukg.modules.ontology.active_ontology import ActiveOntology

logger = get_logger(__name__)

_SPACY_LABEL_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORGANIZATION,
}

# Coarse EntityType -> core OntologyClass name mapping used to set
# Entity.ontology_class_id at creation time. This is intentionally coarse:
# classifying an entity into a precise domain class (e.g. "Model" vs
# "Dataset") would need a real classifier, out of scope for this pass — see
# MODULE_3_5_NOTES.md. Falling back to the core "Concept" class still lets
# triple validation (stage 10) do a meaningful domain/range check for the
# core relations (is_prerequisite_of, is_related_to, part_of), which are the
# majority of what relation extraction produces.
_ENTITY_TYPE_TO_CLASS_NAME: dict[EntityType, str] = {
    EntityType.PERSON: "Person",
    EntityType.FORMULA: "Formula",
    EntityType.ALGORITHM: "Algorithm",
    EntityType.THEOREM: "Theorem",
    EntityType.DEFINITION: "Definition",
}
_DEFAULT_CLASS_NAME = "Concept"

_EXTRACTOR_VERSION = "1.0.0"
_SEARCH_TOP_K = 5


class EntityLinkingService:
    """
    Stage 7: resolves each Mention to a canonical Entity. Two-stage match per
    the pipeline spec's Phase 3 — exact string/alias boost first (cheap,
    precise), then FAISS cosine similarity search over entity embeddings; a
    mention with no sufficiently similar existing entity becomes a new
    canonical Entity. Mentions are embedded in a single batched call per
    invocation (not per-mention) since the embedding provider is a metered
    Azure endpoint.
    """

    def __init__(
        self,
        entity_repository: EntityRepository,
        vector_index: VectorIndexRepository,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float | None = None,
        query_similarity_threshold: float | None = None,
    ) -> None:
        self._entities = entity_repository
        self._index = vector_index
        self._embeddings = embedding_provider
        self._threshold = (
            similarity_threshold if similarity_threshold is not None else get_settings().thresholds.entity_linking_similarity
        )
        # Deliberately lower and separate from _threshold: see
        # ThresholdSettings.entity_linking_query_similarity's docstring for
        # the full rationale (ingestion-time merges are hard to undo,
        # query-time read-only tagging isn't).
        self._query_threshold = (
            query_similarity_threshold
            if query_similarity_threshold is not None
            else get_settings().thresholds.entity_linking_query_similarity
        )

    def link(
        self,
        mentions: list[Mention],
        document_id: UUID,
        active_ontology: ActiveOntology | None,
        *,
        create_if_missing: bool = True,
    ) -> list[Entity]:
        """create_if_missing=False is the read-only mode used for query-time
        concept tagging (see agents/shared/concept_tagger.py): unmatched
        mentions are simply dropped rather than minting new KG entities from
        a live user query. active_ontology is only required (and only used)
        when create_if_missing=True, since it's solely needed to resolve a
        *new* entity's ontology class."""
        if not mentions:
            return []
        if create_if_missing and active_ontology is None:
            raise ValueError("active_ontology is required when create_if_missing=True")

        vectors = self._embeddings.embed([m.surface_form for m in mentions])

        # Keyed by id, not appended as encountered: a later mention can attach
        # to (and mutate — new alias, new mention_id) an entity created by an
        # earlier mention in this same batch, and the caller must see that
        # final state, not the stale pre-mutation object from its first
        # appearance.
        linked_by_id: dict[UUID, Entity] = {}
        for mention, vector in zip(mentions, vectors, strict=True):
            entity = self._resolve(mention, vector, document_id, active_ontology, create_if_missing=create_if_missing)
            if entity is not None:
                linked_by_id[entity.id] = entity

        linked = list(linked_by_id.values())
        logger.info("entity_linking_done", document_id=str(document_id), mention_count=len(mentions), entity_count=len(linked))
        return linked

    def _resolve(
        self,
        mention: Mention,
        vector: list[float],
        document_id: UUID,
        active_ontology: ActiveOntology | None,
        *,
        create_if_missing: bool,
    ) -> Entity | None:
        existing = self._entities.find_by_canonical_name(mention.surface_form)
        if existing is None:
            threshold = self._threshold if create_if_missing else self._query_threshold
            existing = self._best_similarity_match(vector, threshold)

        if existing is not None:
            # create_if_missing doubles as "read-only mode" overall: a
            # query-time tag (ConceptTagger) must not persist any mutation to
            # the KG, including appending its mention_id/alias onto an
            # entity it merely *matched* — only the reviewed ingestion path
            # is allowed to write mention/alias bookkeeping into storage.
            return self._attach_mention(existing, mention, persist=create_if_missing)

        if not create_if_missing:
            return None

        assert active_ontology is not None  # guaranteed by the link() guard above
        return self._create_entity(mention, vector, document_id, active_ontology)

    def _best_similarity_match(self, vector: list[float], threshold: float) -> Entity | None:
        candidates = self._index.search(vector, top_k=_SEARCH_TOP_K)
        for entity_id, score in candidates:
            if score >= threshold:
                return self._entities.get(entity_id)
        return None

    def _attach_mention(self, entity: Entity, mention: Mention, *, persist: bool = True) -> Entity:
        entity.add_alias(mention.surface_form)
        if mention.id not in entity.mention_ids:
            entity.mention_ids.append(mention.id)
        entity.touch()
        if persist:
            self._entities.save(entity)
        return entity

    def _create_entity(
        self, mention: Mention, vector: list[float], document_id: UUID, active_ontology: ActiveOntology
    ) -> Entity:
        entity_type = _SPACY_LABEL_TO_ENTITY_TYPE.get(mention.candidate_type or "", EntityType.CONCEPT)
        class_name = _ENTITY_TYPE_TO_CLASS_NAME.get(entity_type, _DEFAULT_CLASS_NAME)
        ontology_class = active_ontology.get_class(class_name) or active_ontology.get_class(_DEFAULT_CLASS_NAME)

        entity = Entity(
            canonical_name=mention.surface_form,
            entity_type=entity_type,
            ontology_class_id=ontology_class.id if ontology_class is not None else None,
            embedding=vector,
            mention_ids=[mention.id],
            provenance=[
                Provenance(
                    source_document_id=document_id,
                    chunk_id=mention.chunk_id,
                    extractor_name="EntityLinkingService",
                    extractor_version=_EXTRACTOR_VERSION,
                    raw_span_start=mention.span_start,
                    raw_span_end=mention.span_end,
                )
            ],
        )
        self._entities.save(entity)
        self._index.add(entity.id, vector)
        return entity
