from uuid import uuid4

from edukg.core.models import (
    ConfidenceScore,
    Entity,
    OntologyClass,
    OntologyRelation,
    Provenance,
    Relation,
    Subject,
)
from edukg.core.models.enums import EntityType, ValidationStatus
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.triple_generation.triple_generator import TripleGenerator
from edukg.modules.triple_validation.domain_range_validator import TripleValidationService


def _provenance(document_id) -> Provenance:
    return Provenance(source_document_id=document_id, extractor_name="test", extractor_version="1.0.0")


def _relation(subject_id, object_id, predicate: str, relation_id, confidence: float = 0.8, method: str = "cue_phrase") -> Relation:
    return Relation(
        subject_entity_id=subject_id,
        predicate=predicate,
        object_entity_id=object_id,
        confidence=ConfidenceScore(value=confidence, method=method),
        provenance=_provenance(uuid4()),
        ontology_relation_id=relation_id,
    )


def test_triple_generator_projects_relation_fields() -> None:
    subject_id, object_id, relation_id = uuid4(), uuid4(), uuid4()
    relation = _relation(subject_id, object_id, "is_related_to", relation_id)

    triples = TripleGenerator().generate([relation])

    assert len(triples) == 1
    triple = triples[0]
    assert triple.subject_entity_id == subject_id
    assert triple.object_entity_id == object_id
    assert triple.predicate == "is_related_to"
    assert triple.confidence == relation.confidence
    assert triple.relation_id == relation.id
    assert triple.validation_status == ValidationStatus.PENDING


def _build_ontology() -> tuple[ActiveOntology, OntologyClass, OntologyClass, OntologyRelation]:
    model_class = OntologyClass(name="Model", subject=Subject.MACHINE_LEARNING)
    dataset_class = OntologyClass(name="Dataset", subject=Subject.MACHINE_LEARNING)
    trained_on = OntologyRelation(
        name="trained_on", subject=Subject.MACHINE_LEARNING, domain_class_id=model_class.id, range_class_id=dataset_class.id
    )
    active = ActiveOntology(
        subject=Subject.MACHINE_LEARNING,
        version="1.0.0",
        classes_by_name={"Model": model_class, "Dataset": dataset_class},
        relations_by_name={"trained_on": trained_on},
    )
    return active, model_class, dataset_class, trained_on


def test_validator_accepts_triple_matching_domain_and_range() -> None:
    active, model_class, dataset_class, trained_on = _build_ontology()
    subject_entity = Entity(canonical_name="GPT", entity_type=EntityType.CONCEPT, ontology_class_id=model_class.id)
    object_entity = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = TripleGenerator().generate(
        [_relation(subject_entity.id, object_entity.id, "trained_on", trained_on.id, confidence=0.9)]
    )[0]

    result = TripleValidationService().validate(triple, {subject_entity.id: subject_entity, object_entity.id: object_entity}, active)

    assert result.status == ValidationStatus.VALID
    assert result.issues == []


def test_validator_flags_unknown_predicate_as_invalid() -> None:
    active, model_class, dataset_class, _trained_on = _build_ontology()
    subject_entity = Entity(canonical_name="GPT", entity_type=EntityType.CONCEPT, ontology_class_id=model_class.id)
    object_entity = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = TripleGenerator().generate(
        [_relation(subject_entity.id, object_entity.id, "nonexistent_predicate", uuid4(), confidence=0.9)]
    )[0]

    result = TripleValidationService().validate(triple, {subject_entity.id: subject_entity, object_entity.id: object_entity}, active)

    assert result.status == ValidationStatus.INVALID
    assert any(i.code == "UNKNOWN_PREDICATE" for i in result.issues)


def test_validator_warns_on_domain_class_mismatch() -> None:
    active, model_class, dataset_class, trained_on = _build_ontology()
    wrong_subject = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    object_entity = Entity(canonical_name="Common Crawl 2", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = TripleGenerator().generate(
        [_relation(wrong_subject.id, object_entity.id, "trained_on", trained_on.id, confidence=0.9)]
    )[0]

    result = TripleValidationService().validate(triple, {wrong_subject.id: wrong_subject, object_entity.id: object_entity}, active)

    assert result.status == ValidationStatus.WARNING
    assert any(i.code == "DOMAIN_MISMATCH" for i in result.issues)
    assert result.is_valid  # WARNING still passes downstream (fusion) per ValidationResult.is_valid


def test_validator_warns_on_low_confidence() -> None:
    active, model_class, dataset_class, trained_on = _build_ontology()
    subject_entity = Entity(canonical_name="GPT", entity_type=EntityType.CONCEPT, ontology_class_id=model_class.id)
    object_entity = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = TripleGenerator().generate(
        [_relation(subject_entity.id, object_entity.id, "trained_on", trained_on.id, confidence=0.1)]
    )[0]

    result = TripleValidationService().validate(triple, {subject_entity.id: subject_entity, object_entity.id: object_entity}, active)

    assert any(i.code == "LOW_CONFIDENCE" for i in result.issues)
