from pathlib import Path
from uuid import uuid4

from edukg.core.models import Entity, Mention, OntologyClass, Subject
from edukg.core.models.enums import EntityType
from edukg.infrastructure.faiss.faiss_vector_index import FAISSVectorIndexRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.modules.entity_linking.linking_service import EntityLinkingService
from edukg.modules.ontology.active_ontology import ActiveOntology

_DIMENSION = 4


class _FakeEmbeddingProvider:
    """Deterministic, test-controlled embeddings — no real model/network call.
    Vectors are supplied by the test so similarity outcomes are predictable."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def dimension(self) -> int:
        return _DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


def _active_ontology() -> tuple[ActiveOntology, OntologyClass, OntologyClass]:
    concept = OntologyClass(name="Concept", subject=Subject.UNKNOWN, is_core=True)
    person = OntologyClass(name="Person", subject=Subject.UNKNOWN, is_core=True)
    active = ActiveOntology(
        subject=Subject.MACHINE_LEARNING,
        version="1.0.0",
        classes_by_name={"Concept": concept, "Person": person},
    )
    return active, concept, person


def _mention(surface_form: str, span_start: int = 0, candidate_type: str | None = None) -> Mention:
    return Mention(
        chunk_id=uuid4(),
        document_id=uuid4(),
        surface_form=surface_form,
        span_start=span_start,
        span_end=span_start + len(surface_form),
        candidate_type=candidate_type,
    )


def _build_service(tmp_path: Path, vectors: dict[str, list[float]]) -> EntityLinkingService:
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    vector_index = FAISSVectorIndexRepository(dimension=_DIMENSION)
    return EntityLinkingService(entity_repo, vector_index, _FakeEmbeddingProvider(vectors))


def test_similar_surface_forms_link_to_same_entity_via_faiss(tmp_path: Path) -> None:
    active, concept, _person = _active_ontology()
    vectors = {
        "NN": [1.0, 0.0, 0.0, 0.0],
        "Neural Network": [1.0, 0.0, 0.0, 0.0],  # identical vector -> cosine 1.0, well above threshold
        "Photosynthesis": [0.0, 1.0, 0.0, 0.0],  # orthogonal -> must NOT link to the above
    }
    service = _build_service(tmp_path, vectors)
    document_id = uuid4()

    mentions = [_mention("NN", 0), _mention("Neural Network", 10), _mention("Photosynthesis", 30)]
    entities = service.link(mentions, document_id, active)

    assert len(entities) == 2
    nn_entity = next(e for e in entities if e.canonical_name == "NN")
    assert "Neural Network" in nn_entity.aliases
    assert nn_entity.ontology_class_id == concept.id

    photo_entity = next(e for e in entities if e.canonical_name == "Photosynthesis")
    assert photo_entity.id != nn_entity.id


def test_exact_name_match_reuses_entity_without_new_alias(tmp_path: Path) -> None:
    active, _concept, _person = _active_ontology()
    vectors = {"Model": [1.0, 0.0, 0.0, 0.0]}
    service = _build_service(tmp_path, vectors)
    document_id = uuid4()

    mentions = [_mention("Model", 0), _mention("Model", 50)]
    entities = service.link(mentions, document_id, active)

    assert len(entities) == 1
    assert entities[0].aliases == []  # identical name is not re-added as an alias
    assert len(entities[0].mention_ids) == 2


def _seed_dna_entity(tmp_path: Path) -> tuple[FilesystemEntityRepository, FAISSVectorIndexRepository, Entity]:
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    vector_index = FAISSVectorIndexRepository(dimension=_DIMENSION)
    existing = Entity(canonical_name="DNA", entity_type=EntityType.CONCEPT)
    entity_repo.save(existing)
    vector_index.add(existing.id, [1.0, 0.0, 0.0, 0.0])
    return entity_repo, vector_index, existing


# cosine similarity of this vector against [1,0,0,0] ("DNA") is exactly 0.5 —
# a real "genetic material"/"DNA"-style paraphrase score (see
# ThresholdSettings.entity_linking_query_similarity's docstring), chosen to
# sit strictly between the query-time floor (0.4) and the ingestion floor (0.85).
_MODERATE_SIMILARITY_VECTOR = [0.5, 0.8660254, 0.0, 0.0]


def test_ingestion_time_linking_does_not_merge_moderate_similarity_matches(tmp_path: Path) -> None:
    """A moderate-similarity paraphrase must NOT silently merge into an
    existing entity during ingestion (create_if_missing=True, default 0.85
    floor) — a wrong merge there is a hard-to-undo KG mistake."""
    active, _concept, _person = _active_ontology()
    entity_repo, vector_index, _existing = _seed_dna_entity(tmp_path)
    embeddings = _FakeEmbeddingProvider({"genetic material": _MODERATE_SIMILARITY_VECTOR})
    service = EntityLinkingService(entity_repo, vector_index, embeddings)

    result = service.link([_mention("genetic material", 0)], uuid4(), active, create_if_missing=True)

    assert result[0].canonical_name == "genetic material"  # created new, did not merge into DNA
    assert len(entity_repo.list_all()) == 2


def test_query_time_tagging_surfaces_moderate_similarity_matches(tmp_path: Path) -> None:
    """The same moderate-similarity paraphrase SHOULD surface for read-only
    query-time tagging (create_if_missing=False, default 0.4 floor) since it
    never mutates the KG — a student asking about 'genetic material' should
    still surface the existing DNA concept."""
    active, _concept, _person = _active_ontology()
    entity_repo, vector_index, existing = _seed_dna_entity(tmp_path)
    embeddings = _FakeEmbeddingProvider({"genetic material": _MODERATE_SIMILARITY_VECTOR})
    service = EntityLinkingService(entity_repo, vector_index, embeddings)

    tagged = service.link([_mention("genetic material", 0)], uuid4(), active, create_if_missing=False)

    assert [e.id for e in tagged] == [existing.id]
    assert len(entity_repo.list_all()) == 1  # still read-only: no new entity minted
    assert entity_repo.get(existing.id).aliases == []  # and nothing persisted onto the matched entity either


def test_person_candidate_type_resolves_person_ontology_class(tmp_path: Path) -> None:
    active, _concept, person = _active_ontology()
    vectors = {"Ada Lovelace": [0.0, 0.0, 1.0, 0.0]}
    service = _build_service(tmp_path, vectors)
    document_id = uuid4()

    mentions = [_mention("Ada Lovelace", 0, candidate_type="PERSON")]
    entities = service.link(mentions, document_id, active)

    assert entities[0].entity_type.value == "person"
    assert entities[0].ontology_class_id == person.id
