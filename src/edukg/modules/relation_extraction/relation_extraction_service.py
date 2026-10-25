from __future__ import annotations

import itertools
from collections import defaultdict
from uuid import UUID

from edukg.core.logging_setup import get_logger
from edukg.core.models import Chunk, ConfidenceScore, Entity, Mention, Provenance, Relation
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.relation_extraction.cooccurrence_fallback import CooccurrenceFallback
from edukg.modules.relation_extraction.cue_phrase_extractor import CuePhraseExtractor
from edukg.modules.relation_extraction.dependency_relation_extractor import DependencyRelationExtractor
from edukg.modules.relation_extraction.llm_relation_extractor import LLMRelationExtractor

logger = get_logger(__name__)

# Dependency-parse verb lemmas that map onto real ontology relation names, as
# (relation_name, reversed) — see CuePhraseExtractor/CUE_PHRASES's docstring
# for what `reversed` means. "X requires/needs/assumes Y" puts the
# prerequisite (Y) after the dependent concept (X), the same direction issue
# as the matching cue phrases below. Anything not covered here falls through
# to the cue-phrase tier rather than inventing an unresolvable predicate.
VERB_TO_RELATION: dict[str, tuple[str, bool]] = {
    "require": ("is_prerequisite_of", True),
    "need": ("is_prerequisite_of", True),
    "assume": ("is_prerequisite_of", True),
    "define": ("defines", False),
    "minimize": ("minimizes", False),
    "train": ("trained_on", False),
    "use": ("is_related_to", False),
    "relate": ("is_related_to", False),
}

_EXTRACTOR_VERSION = "1.0.0"


class RelationExtractionService:
    """
    Stage 8: for each chunk, pairs up the entities mentioned within it and
    runs the 4-tier cascade (dependency parse -> cue phrase -> LLM judgment
    -> co-occurrence fallback) to assign each pair a predicate, resolved
    against the active ontology's relations. Every candidate pair produces a
    Relation, including weak co-occurrence-only ones — confidence-based
    filtering happens downstream in Knowledge Fusion (stage 11), not here.

    The LLM tier is optional (llm_relation_extractor=None skips it entirely,
    falling straight to co-occurrence) — this keeps every existing caller
    and test that constructs RelationExtractionService(nlp) with no LLM
    working unchanged; it's opted into explicitly where wired in
    pipeline_runner.py.
    """

    def __init__(self, nlp, llm_relation_extractor: LLMRelationExtractor | None = None) -> None:
        self._nlp = nlp
        self._dependency_extractor = DependencyRelationExtractor(nlp)
        self._cue_extractor = CuePhraseExtractor()
        self._llm_extractor = llm_relation_extractor
        self._fallback = CooccurrenceFallback()

    def extract(
        self,
        document_id: UUID,
        chunks: list[Chunk],
        mentions: list[Mention],
        entities: list[Entity],
        active_ontology: ActiveOntology,
    ) -> list[Relation]:
        entities_by_mention_id: dict[UUID, Entity] = {}
        for entity in entities:
            for mention_id in entity.mention_ids:
                entities_by_mention_id[mention_id] = entity

        mentions_by_chunk: dict[UUID, list[Mention]] = defaultdict(list)
        for mention in mentions:
            mentions_by_chunk[mention.chunk_id].append(mention)

        relations: list[Relation] = []
        for chunk in chunks:
            relations.extend(
                self._extract_for_chunk(
                    chunk, mentions_by_chunk.get(chunk.id, []), entities_by_mention_id, active_ontology, document_id
                )
            )

        logger.info("relation_extraction_done", document_id=str(document_id), relation_count=len(relations))
        return relations

    def _extract_for_chunk(
        self,
        chunk: Chunk,
        chunk_mentions: list[Mention],
        entities_by_mention_id: dict[UUID, Entity],
        active_ontology: ActiveOntology,
        document_id: UUID,
    ) -> list[Relation]:
        # One representative (earliest) mention per entity present in this chunk.
        entity_mentions: dict[UUID, Mention] = {}
        for mention in chunk_mentions:
            entity = entities_by_mention_id.get(mention.id)
            if entity is None:
                continue
            existing = entity_mentions.get(entity.id)
            if existing is None or mention.span_start < existing.span_start:
                entity_mentions[entity.id] = mention

        if len(entity_mentions) < 2:
            return []

        doc = self._nlp(chunk.text)
        relations: list[Relation] = []

        # Pairing is scoped to entities that co-occur within the same
        # sentence, not the whole chunk. A 512-token chunk can hold dozens of
        # mentions across unrelated sentences/paragraphs; all-pairs over the
        # whole chunk produces a combinatorial flood of near-meaningless
        # co-occurrence relations that Knowledge Fusion then has to filter
        # back out (observed: 447 relations -> 5 accepted triples on a real
        # document). Same-sentence co-occurrence is both a stronger signal
        # and what DependencyRelationExtractor already requires internally.
        for sent in doc.sents:
            sentence_entities = [
                (entity_id, mention)
                for entity_id, mention in entity_mentions.items()
                if sent.start_char <= mention.span_start < sent.end_char
            ]
            if len(sentence_entities) < 2:
                continue

            ordered = sorted(sentence_entities, key=lambda kv: kv[1].span_start)
            for (entity_a_id, mention_a), (entity_b_id, mention_b) in itertools.combinations(ordered, 2):
                entity_a = entities_by_mention_id.get(mention_a.id)
                entity_b = entities_by_mention_id.get(mention_b.id)
                resolved = self._resolve_predicate(
                    doc, chunk.text, sent.text, mention_a, mention_b, entity_a, entity_b, active_ontology
                )
                if resolved is None:
                    continue  # co-occurrence fallback declined: neither entity has a resolved ontology class
                predicate, method, confidence, reversed_ = resolved
                ontology_relation = active_ontology.get_relation(predicate) or active_ontology.get_relation(
                    CooccurrenceFallback.RELATION_NAME
                )
                if ontology_relation is None:
                    logger.warning("relation_extraction_no_fallback_relation", predicate=predicate)
                    continue

                # mention_a/entity_a_id is always the earlier mention in text
                # order, but that isn't always the relation's grammatical
                # subject — e.g. "X requires Y": Y is is_prerequisite_of's
                # subject even though X appears first. `reversed_` (set by
                # the dependency/cue-phrase tier that resolved this pair)
                # swaps which entity becomes subject vs. object.
                subject_id, object_id = (entity_b_id, entity_a_id) if reversed_ else (entity_a_id, entity_b_id)

                relations.append(
                    Relation(
                        subject_entity_id=subject_id,
                        predicate=ontology_relation.name,
                        object_entity_id=object_id,
                        confidence=ConfidenceScore(value=confidence, method=method),
                        provenance=Provenance(
                            source_document_id=document_id,
                            chunk_id=chunk.id,
                            extractor_name="RelationExtractionService",
                            extractor_version=_EXTRACTOR_VERSION,
                            raw_span_start=mention_a.span_start,
                            raw_span_end=mention_b.span_end,
                        ),
                        ontology_relation_id=ontology_relation.id,
                    )
                )
        return relations

    def _resolve_predicate(
        self,
        doc,
        chunk_text: str,
        sentence_text: str,
        mention_a: Mention,
        mention_b: Mention,
        entity_a: Entity | None,
        entity_b: Entity | None,
        active_ontology: ActiveOntology,
    ) -> tuple[str, str, float, bool] | None:
        span_a = (mention_a.span_start, mention_a.span_end)
        span_b = (mention_b.span_start, mention_b.span_end)

        verb_lemma = self._dependency_extractor.extract(doc, span_a, span_b)
        if verb_lemma is not None:
            mapped = VERB_TO_RELATION.get(verb_lemma)
            if mapped is not None and active_ontology.has_relation(mapped[0]):
                relation_name, reversed_ = mapped
                return relation_name, "dependency_parse", 0.7, reversed_

        gap_start, gap_end = mention_a.span_end, mention_b.span_start
        gap_text = chunk_text[gap_start:gap_end] if gap_end > gap_start else ""
        cue_result = self._cue_extractor.extract(gap_text)
        if cue_result is not None and active_ontology.has_relation(cue_result[0]):
            relation_name, reversed_ = cue_result
            return relation_name, "cue_phrase", 0.75, reversed_

        # Tier 3: ask the LLM directly, but only for a pair where both
        # entities already resolved to a real ontology class — same
        # ontology-verification gate CooccurrenceFallback requires below, for
        # the same reason: there's no point spending an LLM call judging a
        # relation for an entity the KG can't reason about as a concept yet.
        if (
            self._llm_extractor is not None
            and entity_a is not None
            and entity_b is not None
            and entity_a.ontology_class_id is not None
            and entity_b.ontology_class_id is not None
        ):
            candidate_relations = [r.name for r in active_ontology.relations]
            llm_result = self._llm_extractor.extract(sentence_text, entity_a.canonical_name, entity_b.canonical_name, candidate_relations)
            if llm_result is not None:
                relation_name, reversed_ = llm_result
                if active_ontology.has_relation(relation_name):
                    return relation_name, "llm", 0.6, reversed_

        fallback_relation = self._fallback.extract(entity_a, entity_b)
        if fallback_relation is None:
            return None
        return fallback_relation, "co_occurrence", 0.35, False
