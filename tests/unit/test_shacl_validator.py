from uuid import uuid4

from edukg.core.models import ConfidenceScore, Entity, OntologyClass, OntologyRelation, Provenance, Subject, Triple
from edukg.core.models.enums import EntityType, ValidationStatus
from edukg.modules.ontology.active_ontology import ActiveOntology
from edukg.modules.triple_validation.shacl_validator import validate_triple


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


def _triple(subject_id, predicate, object_id, relation_id) -> Triple:
    return Triple(
        subject_entity_id=subject_id,
        predicate=predicate,
        object_entity_id=object_id,
        confidence=ConfidenceScore(value=0.9, method="test"),
        provenance=Provenance(source_document_id=uuid4(), extractor_name="test", extractor_version="1.0.0"),
        source_document_id=uuid4(),
        extractor_version="1.0.0",
        relation_id=relation_id,
        validation_status=ValidationStatus.PENDING,
    )


def test_validate_triple_finds_no_issues_for_a_conforming_triple() -> None:
    active, model_class, dataset_class, trained_on = _build_ontology()
    subject_entity = Entity(canonical_name="GPT", entity_type=EntityType.CONCEPT, ontology_class_id=model_class.id)
    object_entity = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = _triple(subject_entity.id, "trained_on", object_entity.id, trained_on.id)

    issues = validate_triple(triple, subject_entity, object_entity, active)

    assert issues == []


def test_validate_triple_flags_a_domain_range_violation() -> None:
    active, model_class, dataset_class, trained_on = _build_ontology()
    # Subject typed as Dataset, not Model — violates trained_on's domain constraint.
    wrong_subject = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    object_entity = Entity(canonical_name="Common Crawl 2", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = _triple(wrong_subject.id, "trained_on", object_entity.id, trained_on.id)

    issues = validate_triple(triple, wrong_subject, object_entity, active)

    assert len(issues) == 1
    assert issues[0].code == "SHACL_DOMAIN_RANGE_VIOLATION"
    assert issues[0].severity == ValidationStatus.WARNING


def test_validate_triple_skips_entities_with_no_resolved_class() -> None:
    active, model_class, dataset_class, trained_on = _build_ontology()
    untyped_subject = Entity(canonical_name="Something", entity_type=EntityType.CONCEPT, ontology_class_id=None)
    object_entity = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = _triple(untyped_subject.id, "trained_on", object_entity.id, trained_on.id)

    issues = validate_triple(triple, untyped_subject, object_entity, active)

    assert issues == []


def test_validate_triple_skips_an_unknown_predicate() -> None:
    active, model_class, dataset_class, _trained_on = _build_ontology()
    subject_entity = Entity(canonical_name="GPT", entity_type=EntityType.CONCEPT, ontology_class_id=model_class.id)
    object_entity = Entity(canonical_name="Common Crawl", entity_type=EntityType.CONCEPT, ontology_class_id=dataset_class.id)
    triple = _triple(subject_entity.id, "nonexistent_predicate", object_entity.id, uuid4())

    issues = validate_triple(triple, subject_entity, object_entity, active)

    assert issues == []
