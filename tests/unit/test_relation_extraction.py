import json
from uuid import uuid4

from edukg.core.models import Chunk, Entity, Mention, OntologyClass, OntologyRelation, Subject
from edukg.core.models.enums import EntityType, ExtractionMethod
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.relation_extraction.cooccurrence_fallback import CooccurrenceFallback
from edukg.modules.relation_extraction.cue_phrase_extractor import CuePhraseExtractor
from edukg.modules.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from edukg.modules.relation_extraction.relation_extraction_service import RelationExtractionService


class _FakeLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_call: dict | None = None

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        self.last_call = {"system_prompt": system_prompt, "user_prompt": user_prompt}
        return self.response

    def model_name(self) -> str:
        return "fake-model"


class _FakeSent:
    def __init__(self, text: str, start_char: int, end_char: int) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class _FakeDoc:
    """Stands in for a parsed spaCy Doc whose dependency tree yields nothing,
    so tests exercise the cue-phrase/co-occurrence tiers deterministically
    without requiring the real en_core_web_sm model. Treats the whole text as
    a single sentence, matching the single-sentence fixtures these tests use."""

    def __init__(self, text: str) -> None:
        self.sents = [_FakeSent(text, 0, len(text))]

    def char_span(self, start: int, end: int, alignment_mode: str = "strict"):
        return None


class _FakeNLP:
    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc(text)


def test_cue_phrase_extractor_matches_known_phrase() -> None:
    extractor = CuePhraseExtractor()
    # "requires" puts the prerequisite *after* the dependent concept in text
    # order ("X requires Y" -- Y is the prerequisite), unlike "is a
    # prerequisite of" which already reads subject-first -- reversed=True
    # tells the caller to swap subject/object before storing the Relation.
    assert extractor.extract(" requires ") == ("is_prerequisite_of", True)
    assert extractor.extract(" is a prerequisite of ") == ("is_prerequisite_of", False)
    assert extractor.extract(" builds on ") == ("is_prerequisite_of", True)
    assert extractor.extract(" assumes knowledge of ") == ("is_prerequisite_of", True)
    assert extractor.extract(" consists of ") == ("part_of", True)
    assert extractor.extract(" is unrelated filler ") is None


def test_cooccurrence_fallback_returns_is_related_to_when_called_without_entities() -> None:
    # No entities passed (the ontology-verification gate is opt-in) -- same
    # as any caller that hasn't been updated to pass entities yet.
    assert CooccurrenceFallback().extract() == "is_related_to"


def test_cooccurrence_fallback_returns_is_related_to_when_both_entities_are_ontology_verified() -> None:
    class_id = uuid4()
    entity_a = Entity(canonical_name="A", entity_type=EntityType.CONCEPT, ontology_class_id=class_id)
    entity_b = Entity(canonical_name="B", entity_type=EntityType.CONCEPT, ontology_class_id=class_id)
    assert CooccurrenceFallback().extract(entity_a, entity_b) == "is_related_to"


def test_llm_relation_extractor_returns_a_valid_relation() -> None:
    llm = _FakeLLMProvider(json.dumps({"relation": "is_prerequisite_of", "subject_is_first": True}))
    extractor = LLMRelationExtractor(llm)

    result = extractor.extract("Probability underlies gradient descent.", "Probability", "Gradient Descent", ["is_prerequisite_of", "is_related_to"])

    assert result == ("is_prerequisite_of", False)
    assert "is_prerequisite_of" in llm.last_call["system_prompt"]


def test_llm_relation_extractor_handles_subject_is_second() -> None:
    llm = _FakeLLMProvider(json.dumps({"relation": "is_prerequisite_of", "subject_is_first": False}))
    extractor = LLMRelationExtractor(llm)

    result = extractor.extract("Gradient descent relies on probability.", "Gradient Descent", "Probability", ["is_prerequisite_of"])

    assert result == ("is_prerequisite_of", True)  # entity_a (first-named) is NOT the subject here


def test_llm_relation_extractor_returns_none_when_no_relation_applies() -> None:
    llm = _FakeLLMProvider(json.dumps({"relation": None, "subject_is_first": True}))
    extractor = LLMRelationExtractor(llm)

    assert extractor.extract("Unrelated sentence.", "A", "B", ["is_prerequisite_of"]) is None


def test_llm_relation_extractor_degrades_to_none_on_malformed_response() -> None:
    extractor = LLMRelationExtractor(_FakeLLMProvider("not json at all"))
    assert extractor.extract("text", "A", "B", ["is_prerequisite_of"]) is None


def test_llm_relation_extractor_rejects_a_relation_outside_the_candidate_list() -> None:
    # Guards against the LLM inventing a relation name not on the ontology's
    # list, despite being told not to -- never trust it blindly.
    llm = _FakeLLMProvider(json.dumps({"relation": "made_up_relation", "subject_is_first": True}))
    extractor = LLMRelationExtractor(llm)

    assert extractor.extract("text", "A", "B", ["is_prerequisite_of"]) is None


def test_llm_relation_extractor_degrades_to_none_when_llm_call_raises() -> None:
    class _RaisingLLMProvider:
        def complete_json(self, **kwargs):
            raise RuntimeError("Azure OpenAI unreachable")

        def model_name(self) -> str:
            return "fake-model"

    extractor = LLMRelationExtractor(_RaisingLLMProvider())
    assert extractor.extract("text", "A", "B", ["is_prerequisite_of"]) is None


def test_cooccurrence_fallback_declines_when_either_entity_is_not_ontology_verified() -> None:
    # Regression test: pairing any two co-occurring mentions regardless of
    # ontology status produced weak, meaningless is_related_to edges for
    # entities that never resolved to a real concept in the first place.
    verified = Entity(canonical_name="A", entity_type=EntityType.CONCEPT, ontology_class_id=uuid4())
    unverified = Entity(canonical_name="B", entity_type=EntityType.CONCEPT, ontology_class_id=None)
    assert CooccurrenceFallback().extract(verified, unverified) is None
    assert CooccurrenceFallback().extract(unverified, verified) is None
    assert CooccurrenceFallback().extract(unverified, unverified) is None


def _build_active_ontology() -> tuple[ActiveOntology, OntologyClass]:
    concept = OntologyClass(name="Concept", subject=Subject.UNKNOWN, is_core=True)
    prereq = OntologyRelation(
        name="is_prerequisite_of", subject=Subject.UNKNOWN, domain_class_id=concept.id, range_class_id=concept.id
    )
    related = OntologyRelation(
        name="is_related_to", subject=Subject.UNKNOWN, domain_class_id=concept.id, range_class_id=concept.id
    )
    active = ActiveOntology(
        subject=Subject.MACHINE_LEARNING,
        version="1.0.0",
        classes_by_name={"Concept": concept},
        relations_by_name={"is_prerequisite_of": prereq, "is_related_to": related},
    )
    return active, concept


def _entity(name: str, mention_id, ontology_class_id=None) -> Entity:
    return Entity(canonical_name=name, entity_type=EntityType.CONCEPT, mention_ids=[mention_id], ontology_class_id=ontology_class_id)


def test_cue_phrase_tier_resolves_relation_between_entities() -> None:
    # Regression test: "X requires Y" means Y is the prerequisite (must be
    # understood before X), even though X appears first in the text — the
    # cascade previously always assigned subject=first-mention regardless of
    # which entity was grammatically the prerequisite, storing this exact
    # sentence backwards ("Gradient descent is_prerequisite_of Loss
    # function", i.e. claiming gradient descent must be learned *before*
    # loss functions — wrong). Confirmed correct direction: Loss function
    # (the prerequisite) is the subject, Gradient descent (the dependent
    # concept) is the object.
    text = "Gradient descent requires Loss function knowledge."
    idx_a = text.index("Gradient descent")
    idx_b = text.index("Loss function")

    chunk = Chunk(
        document_id=uuid4(),
        sequence_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=10,
    )
    mention_a = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Gradient descent",
        span_start=idx_a, span_end=idx_a + len("Gradient descent"),
    )
    mention_b = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Loss function",
        span_start=idx_b, span_end=idx_b + len("Loss function"),
    )
    entity_a = _entity("Gradient descent", mention_a.id)
    entity_b = _entity("Loss function", mention_b.id)
    active, _concept = _build_active_ontology()

    service = RelationExtractionService(_FakeNLP())
    relations = service.extract(uuid4(), [chunk], [mention_a, mention_b], [entity_a, entity_b], active)

    assert len(relations) == 1
    relation = relations[0]
    assert relation.predicate == "is_prerequisite_of"
    assert relation.confidence.method == "cue_phrase"
    assert relation.subject_entity_id == entity_b.id  # Loss function (the prerequisite)
    assert relation.object_entity_id == entity_a.id  # Gradient descent (the dependent concept)


def test_cue_phrase_tier_keeps_subject_first_order_when_phrase_already_reads_that_way() -> None:
    text = "Loss function is a prerequisite of Gradient descent."
    idx_a = text.index("Loss function")
    idx_b = text.index("Gradient descent")

    chunk = Chunk(
        document_id=uuid4(), sequence_index=0, text=text, char_start=0, char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=10,
    )
    mention_a = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Loss function",
        span_start=idx_a, span_end=idx_a + len("Loss function"),
    )
    mention_b = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Gradient descent",
        span_start=idx_b, span_end=idx_b + len("Gradient descent"),
    )
    entity_a = _entity("Loss function", mention_a.id)
    entity_b = _entity("Gradient descent", mention_b.id)
    active, _concept = _build_active_ontology()

    service = RelationExtractionService(_FakeNLP())
    relations = service.extract(uuid4(), [chunk], [mention_a, mention_b], [entity_a, entity_b], active)

    assert len(relations) == 1
    assert relations[0].subject_entity_id == entity_a.id  # Loss function, already first and already the prerequisite
    assert relations[0].object_entity_id == entity_b.id


def test_no_signal_falls_back_to_weak_co_occurrence_relation() -> None:
    text = "Backpropagation and momentum appear in the same sentence together."
    idx_a = text.index("Backpropagation")
    idx_b = text.index("momentum")

    chunk = Chunk(
        document_id=uuid4(),
        sequence_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=10,
    )
    mention_a = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Backpropagation",
        span_start=idx_a, span_end=idx_a + len("Backpropagation"),
    )
    mention_b = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="momentum",
        span_start=idx_b, span_end=idx_b + len("momentum"),
    )
    active, concept = _build_active_ontology()
    # Ontology-verified (has a resolved class) -- otherwise the co-occurrence
    # fallback now correctly declines to pair them at all (see the
    # declines-when-unverified test below), which would make this fixture
    # test the wrong thing.
    entity_a = _entity("Backpropagation", mention_a.id, ontology_class_id=concept.id)
    entity_b = _entity("momentum", mention_b.id, ontology_class_id=concept.id)

    service = RelationExtractionService(_FakeNLP())
    relations = service.extract(uuid4(), [chunk], [mention_a, mention_b], [entity_a, entity_b], active)

    assert len(relations) == 1
    assert relations[0].predicate == "is_related_to"
    assert relations[0].confidence.method == "co_occurrence"
    assert relations[0].confidence.value < 0.5


def test_unverified_entity_co_occurrence_produces_no_relation() -> None:
    # Regression test: two mentions that merely co-occur in a sentence, where
    # at least one never resolved to a real ontology class, used to still
    # produce a weak is_related_to edge regardless -- pure noise for an
    # entity the KG can't actually reason about as a concept yet.
    text = "Backpropagation and the appendix appear in the same sentence together."
    idx_a = text.index("Backpropagation")
    idx_b = text.index("appendix")

    chunk = Chunk(
        document_id=uuid4(), sequence_index=0, text=text, char_start=0, char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=10,
    )
    mention_a = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Backpropagation",
        span_start=idx_a, span_end=idx_a + len("Backpropagation"),
    )
    mention_b = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="appendix",
        span_start=idx_b, span_end=idx_b + len("appendix"),
    )
    active, concept = _build_active_ontology()
    entity_a = _entity("Backpropagation", mention_a.id, ontology_class_id=concept.id)  # verified
    entity_b = _entity("appendix", mention_b.id, ontology_class_id=None)  # never resolved to a class

    service = RelationExtractionService(_FakeNLP())
    relations = service.extract(uuid4(), [chunk], [mention_a, mention_b], [entity_a, entity_b], active)

    assert relations == []


def test_service_uses_llm_tier_when_wired_and_deterministic_tiers_find_nothing() -> None:
    # No cue phrase, no dependency signal (the fake doc's dependency tree is
    # always empty) -- the LLM tier is the only thing that can resolve this
    # pair to anything other than bare co-occurrence.
    text = "Probability underlies Gradient Descent in this course."
    idx_a = text.index("Probability")
    idx_b = text.index("Gradient Descent")

    chunk = Chunk(
        document_id=uuid4(), sequence_index=0, text=text, char_start=0, char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=10,
    )
    mention_a = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Probability",
        span_start=idx_a, span_end=idx_a + len("Probability"),
    )
    mention_b = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Gradient Descent",
        span_start=idx_b, span_end=idx_b + len("Gradient Descent"),
    )
    active, concept = _build_active_ontology()
    entity_a = _entity("Probability", mention_a.id, ontology_class_id=concept.id)
    entity_b = _entity("Gradient Descent", mention_b.id, ontology_class_id=concept.id)

    llm = _FakeLLMProvider(json.dumps({"relation": "is_prerequisite_of", "subject_is_first": True}))
    service = RelationExtractionService(_FakeNLP(), llm_relation_extractor=LLMRelationExtractor(llm))
    relations = service.extract(uuid4(), [chunk], [mention_a, mention_b], [entity_a, entity_b], active)

    assert len(relations) == 1
    assert relations[0].predicate == "is_prerequisite_of"
    assert relations[0].confidence.method == "llm"
    assert relations[0].confidence.value == 0.6
    assert relations[0].subject_entity_id == entity_a.id  # Probability, the first-named concept


def test_service_falls_back_to_co_occurrence_when_no_llm_extractor_wired() -> None:
    # Regression guard: RelationExtractionService(nlp) with no second arg
    # (every existing caller/test before this round) must keep working --
    # the LLM tier is opt-in, not a breaking change.
    text = "Probability underlies Gradient Descent in this course."
    idx_a = text.index("Probability")
    idx_b = text.index("Gradient Descent")

    chunk = Chunk(
        document_id=uuid4(), sequence_index=0, text=text, char_start=0, char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=10,
    )
    mention_a = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Probability",
        span_start=idx_a, span_end=idx_a + len("Probability"),
    )
    mention_b = Mention(
        chunk_id=chunk.id, document_id=chunk.document_id, surface_form="Gradient Descent",
        span_start=idx_b, span_end=idx_b + len("Gradient Descent"),
    )
    active, concept = _build_active_ontology()
    entity_a = _entity("Probability", mention_a.id, ontology_class_id=concept.id)
    entity_b = _entity("Gradient Descent", mention_b.id, ontology_class_id=concept.id)

    service = RelationExtractionService(_FakeNLP())  # no llm_relation_extractor
    relations = service.extract(uuid4(), [chunk], [mention_a, mention_b], [entity_a, entity_b], active)

    assert len(relations) == 1
    assert relations[0].confidence.method == "co_occurrence"


def test_single_entity_in_chunk_produces_no_relations() -> None:
    chunk = Chunk(
        document_id=uuid4(), sequence_index=0, text="Only one concept here.", char_start=0, char_end=23,
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=5,
    )
    mention = Mention(chunk_id=chunk.id, document_id=chunk.document_id, surface_form="concept", span_start=9, span_end=16)
    entity = _entity("concept", mention.id)
    active, _concept = _build_active_ontology()

    service = RelationExtractionService(_FakeNLP())
    relations = service.extract(uuid4(), [chunk], [mention], [entity], active)

    assert relations == []
