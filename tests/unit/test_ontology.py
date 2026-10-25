from pathlib import Path
from uuid import uuid4

import pytest

from edukg.core.exceptions import OntologyError
from edukg.core.models import (
    ConfidenceScore,
    OntologyClass,
    OntologyExtension,
    OntologyExtensionKind,
    OntologyRelation,
    Subject,
    ValidationStatus,
)
from edukg.infrastructure.cache.filesystem_cache import FilesystemCacheRepository
from edukg.infrastructure.filesystem.ontology_repository import FilesystemOntologyRepository
from edukg.modules.ontology.core_ontology_loader import CoreOntologyLoader
from edukg.modules.ontology.domain_ontology_loader import DomainOntologyLoader
from edukg.modules.ontology.ontology_manager import OntologyManager
from edukg.modules.ontology.ontology_merge_service import OntologyMergeService
from edukg.modules.ontology.ontology_review_service import OntologyReviewService
from edukg.modules.ontology.ontology_validator import OntologyValidationService


def _seed_repository(base_dir: Path) -> FilesystemOntologyRepository:
    repo = FilesystemOntologyRepository(base_dir=base_dir)
    concept = OntologyClass(name="Concept", subject=Subject.UNKNOWN, is_core=True)
    repo.add_class(concept)
    prereq = OntologyRelation(
        name="is_prerequisite_of", subject=Subject.UNKNOWN, domain_class_id=concept.id, range_class_id=concept.id
    )
    repo.add_relation(prereq)
    domain_class = OntologyClass(name="Model", subject=Subject.MACHINE_LEARNING, parent_class_id=concept.id)
    domain_path = repo._domain_path(Subject.MACHINE_LEARNING)
    repo._write_bundle(domain_path, [domain_class], [])
    return repo, concept, prereq, domain_class


def _make_manager(tmp_path: Path):
    repo, concept, prereq, domain_class = _seed_repository(tmp_path / "ontologies")
    cache = FilesystemCacheRepository(base_dir=tmp_path / "cache")
    manager = OntologyManager(repo, cache)
    return manager, repo, cache, concept, prereq, domain_class


def test_core_loader_raises_on_empty_core(tmp_path: Path) -> None:
    repo = FilesystemOntologyRepository(base_dir=tmp_path / "empty")
    loader = CoreOntologyLoader(repo)
    with pytest.raises(OntologyError):
        loader.load()


def test_domain_loader_handles_missing_subject_gracefully(tmp_path: Path) -> None:
    repo = FilesystemOntologyRepository(base_dir=tmp_path / "ontologies")
    loader = DomainOntologyLoader(repo)
    classes, relations = loader.load(Subject.PHYSICS)
    assert classes == []
    assert relations == []


def test_ontology_manager_builds_active_ontology(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)

    assert active.has_class("Concept")  # core class present
    assert active.has_class("Model")  # domain class present
    assert active.has_relation("is_prerequisite_of")
    assert active.version == "1.0.0"


def test_get_active_ontology_dedups_a_domain_class_that_collides_with_a_core_class_name(tmp_path: Path) -> None:
    # Regression test: a domain ontology defining a class with the same name
    # as a core class (e.g. "Concept") used to silently evict the core
    # class via plain dict overwrite in get_active_ontology — any existing
    # parent_class_id/domain_class_id/range_class_id reference to the core
    # class's UUID would then point at nothing reachable by name. Found via
    # a cross-check against a teammate's independently-built codebase.
    repo, concept, prereq, domain_class = _seed_repository(tmp_path / "ontologies")
    duplicate_concept = OntologyClass(
        name="Concept", subject=Subject.MACHINE_LEARNING, aliases=["Idea"], description="duplicate from domain file"
    )
    domain_path = repo._domain_path(Subject.MACHINE_LEARNING)
    repo._write_bundle(domain_path, [domain_class, duplicate_concept], [])
    cache = FilesystemCacheRepository(base_dir=tmp_path / "cache")
    manager = OntologyManager(repo, cache)

    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)

    kept = active.get_class("Concept")
    assert kept is not None
    assert kept.id == concept.id  # the original core class's id was kept, not the domain duplicate's
    assert kept.is_core is True
    assert "Idea" in kept.aliases  # the duplicate's alias was merged in, not lost
    # The core relation that referenced the core Concept class's id must
    # still resolve to the SAME id that survived the merge above — this is
    # what "remapping" means in practice (trivially true here since the core
    # class won and nothing needed remapping, but confirms no corruption).
    prereq_after = active.get_relation("is_prerequisite_of")
    assert prereq_after is not None
    assert prereq_after.domain_class_id == kept.id
    assert prereq_after.range_class_id == kept.id


def test_merge_service_class_returns_existing_instead_of_creating_a_duplicate(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    first = merge_service.merge(extension, active)
    second_extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    second = merge_service.merge(second_extension, manager.get_active_ontology(Subject.MACHINE_LEARNING, force_reload=True))

    assert first.id == second.id  # same class returned, not a second "Dataset" with a new id


def test_merge_service_relation_returns_existing_instead_of_creating_a_duplicate(tmp_path: Path) -> None:
    # Within the SAME subject: re-approving an extension proposing a relation
    # name that was already merged in must return the existing relation, not
    # create a second one under a new id. (A cross-subject collision, e.g. a
    # domain extension re-proposing a core relation name, is a different
    # case — handled by get_active_ontology's merge logic, covered by
    # test_get_active_ontology_dedups_a_domain_class_that_collides_with_a_core_class_name's
    # sibling reasoning, not by this per-subject repository check.)
    manager, repo, cache, concept, prereq, domain_class = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)

    extension = _make_extension(
        OntologyExtensionKind.RELATION,
        proposed_name="trained_on",
        proposed_domain_class_id=domain_class.id,
        proposed_range_class_id=concept.id,
    )
    first = merge_service.merge(extension, active)
    second_extension = _make_extension(
        OntologyExtensionKind.RELATION,
        proposed_name="trained_on",
        proposed_domain_class_id=domain_class.id,
        proposed_range_class_id=concept.id,
    )
    second = merge_service.merge(second_extension, manager.get_active_ontology(Subject.MACHINE_LEARNING, force_reload=True))

    assert first.id == second.id


def test_get_active_ontology_dedups_a_domain_relation_that_collides_with_a_core_relation_name(tmp_path: Path) -> None:
    # Cross-subject case: a domain-subject relation with the same name as a
    # core relation (however it got there — including bypassing
    # OntologyMergeService entirely) must not fracture the active ontology
    # into two same-named relations with different ids.
    repo, concept, prereq, domain_class = _seed_repository(tmp_path / "ontologies")
    duplicate_prereq = OntologyRelation(
        name="is_prerequisite_of", subject=Subject.MACHINE_LEARNING, domain_class_id=domain_class.id, range_class_id=concept.id
    )
    domain_path = repo._domain_path(Subject.MACHINE_LEARNING)
    repo._write_bundle(domain_path, [domain_class], [duplicate_prereq])
    cache = FilesystemCacheRepository(base_dir=tmp_path / "cache")
    manager = OntologyManager(repo, cache)

    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)

    kept = active.get_relation("is_prerequisite_of")
    assert kept is not None
    assert kept.id == prereq.id  # the core relation's id won, not the domain duplicate's


def test_ontology_manager_version_bumps_on_add_class(tmp_path: Path) -> None:
    manager, repo, cache, concept, prereq, domain_class = _make_manager(tmp_path)
    before = manager.current_version(Subject.MACHINE_LEARNING)

    new_class = OntologyClass(name="Dataset", subject=Subject.MACHINE_LEARNING, parent_class_id=concept.id)
    manager.add_class(new_class)

    after = manager.current_version(Subject.MACHINE_LEARNING)
    assert before != after

    active = manager.get_active_ontology(Subject.MACHINE_LEARNING, force_reload=True)
    assert active.has_class("Dataset")


def _make_extension(kind: OntologyExtensionKind, **overrides) -> OntologyExtension:
    defaults = dict(
        kind=kind,
        subject=Subject.MACHINE_LEARNING,
        proposed_name="Dataset",
        justification="Needed to describe training data",
        confidence=ConfidenceScore(value=0.9, method="llm_judge"),
        generator_model="test-model",
    )
    defaults.update(overrides)
    return OntologyExtension(**defaults)


def test_validator_flags_duplicate_class(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    validator = OntologyValidationService()

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Model")  # already exists
    result = validator.validate(extension, active)

    assert result.status == ValidationStatus.INVALID
    assert any(i.code == "DUPLICATE_CLASS" for i in result.issues)


def test_validator_accepts_novel_class(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    validator = OntologyValidationService()

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    result = validator.validate(extension, active)

    assert result.status == ValidationStatus.VALID
    assert result.issues == []


def test_validator_flags_relation_missing_domain_range(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    validator = OntologyValidationService()

    extension = _make_extension(OntologyExtensionKind.RELATION, proposed_name="trained_on")
    result = validator.validate(extension, active)

    assert result.status == ValidationStatus.INVALID
    codes = {i.code for i in result.issues}
    assert "MISSING_DOMAIN" in codes and "MISSING_RANGE" in codes


def test_merge_service_creates_new_class(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    merged = merge_service.merge(extension, active)

    assert merged.name == "Dataset"
    active_after = manager.get_active_ontology(Subject.MACHINE_LEARNING, force_reload=True)
    assert active_after.has_class("Dataset")


def test_merge_service_relation_requires_domain_range(tmp_path: Path) -> None:
    manager, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)

    extension = _make_extension(OntologyExtensionKind.RELATION, proposed_name="trained_on")
    with pytest.raises(OntologyError):
        merge_service.merge(extension, active)


def test_merge_service_alias_attaches_to_parent(tmp_path: Path) -> None:
    manager, repo, cache, concept, prereq, domain_class = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)

    extension = _make_extension(
        OntologyExtensionKind.ALIAS, proposed_name="ML Model", proposed_parent_class_id=domain_class.id
    )
    merged = merge_service.merge(extension, active)

    assert "ML Model" in merged.aliases
    assert merged.id == domain_class.id


def test_review_service_approve_merges_extension(tmp_path: Path) -> None:
    manager, repo, cache, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)
    review_service = OntologyReviewService(repo, merge_service)

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    merged = review_service.approve(extension, active, reviewer_id="teacher-1")

    assert merged.name == "Dataset"


def test_review_service_reject_does_not_merge(tmp_path: Path) -> None:
    manager, repo, cache, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    merge_service = OntologyMergeService(manager)
    review_service = OntologyReviewService(repo, merge_service)

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    decision = review_service.reject(extension, reviewer_id="teacher-1", comment="Not needed")

    assert decision.action.value == "reject"
    active_after = manager.get_active_ontology(Subject.MACHINE_LEARNING, force_reload=True)
    assert not active_after.has_class("Dataset")


def test_review_service_auto_review_approves_valid_high_confidence(tmp_path: Path) -> None:
    manager, repo, cache, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    validator = OntologyValidationService()
    merge_service = OntologyMergeService(manager)
    review_service = OntologyReviewService(repo, merge_service)

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Dataset")
    result = validator.validate(extension, active)

    approved, merged = review_service.auto_review(extension, result, active)

    assert approved is True
    assert merged is not None


def test_review_service_auto_review_leaves_invalid_pending(tmp_path: Path) -> None:
    manager, repo, cache, *_ = _make_manager(tmp_path)
    active = manager.get_active_ontology(Subject.MACHINE_LEARNING)
    validator = OntologyValidationService()
    merge_service = OntologyMergeService(manager)
    review_service = OntologyReviewService(repo, merge_service)

    extension = _make_extension(OntologyExtensionKind.CLASS, proposed_name="Model")  # duplicate -> INVALID
    result = validator.validate(extension, active)

    approved, merged = review_service.auto_review(extension, result, active)

    assert approved is False
    assert merged is None
