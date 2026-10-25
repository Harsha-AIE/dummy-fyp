from pathlib import Path
from uuid import uuid4

from edukg.agents.shared.concept_tagger import ConceptTagger
from edukg.core.models.enums import EntityType
from edukg.core.models import Entity
from edukg.infrastructure.faiss.faiss_vector_index import FAISSVectorIndexRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository
from edukg.modules.entity_extraction.mention_extractor import MentionExtractor
from edukg.modules.entity_linking.linking_service import EntityLinkingService

_DIMENSION = 4


class _FakeEnt:
    def __init__(self, text: str, start_char: int, end_char: int) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = None


class _FakeDoc:
    def __init__(self, ents) -> None:
        self.ents = ents


class _FakeNLP:
    def __init__(self, ents) -> None:
        self._ents = ents

    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc(self._ents)


class _EmptyKeyphraseExtractor:
    def extract_keywords(self, text: str, **kwargs):
        return []


class _FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def dimension(self) -> int:
        return _DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


def test_tag_resolves_existing_entity_without_creating_a_new_one(tmp_path: Path) -> None:
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    vector_index = FAISSVectorIndexRepository(dimension=_DIMENSION)
    existing = Entity(canonical_name="Genome Sequencing", entity_type=EntityType.CONCEPT)
    entity_repo.save(existing)
    vector_index.add(existing.id, [1.0, 0.0, 0.0, 0.0])

    text = "Genome Sequencing helps discover new antibiotics."
    idx = text.index("Genome Sequencing")
    nlp = _FakeNLP([_FakeEnt("Genome Sequencing", idx, idx + len("Genome Sequencing"))])
    mention_extractor = MentionExtractor(nlp, _EmptyKeyphraseExtractor(), _EmptyKeyphraseExtractor())
    embeddings = _FakeEmbeddingProvider({"Genome Sequencing": [1.0, 0.0, 0.0, 0.0]})
    linking_service = EntityLinkingService(entity_repo, vector_index, embeddings)

    tagger = ConceptTagger(mention_extractor, linking_service)
    tagged = tagger.tag(text)

    assert len(tagged) == 1
    assert tagged[0].id == existing.id
    assert len(entity_repo.list_all()) == 1  # no new entity minted
    # A query-time tag must not persist a mutation onto the matched entity
    # either (no appended mention_id from this ephemeral query) — read-only
    # really means read-only, not just "no new entities."
    reloaded = entity_repo.get(existing.id)
    assert reloaded.mention_ids == []


def test_tag_drops_unmatched_terms_instead_of_creating_entities(tmp_path: Path) -> None:
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    vector_index = FAISSVectorIndexRepository(dimension=_DIMENSION)

    text = "Unknown Concept appears here."
    idx = text.index("Unknown Concept")
    nlp = _FakeNLP([_FakeEnt("Unknown Concept", idx, idx + len("Unknown Concept"))])
    mention_extractor = MentionExtractor(nlp, _EmptyKeyphraseExtractor(), _EmptyKeyphraseExtractor())
    embeddings = _FakeEmbeddingProvider({"Unknown Concept": [0.0, 1.0, 0.0, 0.0]})
    linking_service = EntityLinkingService(entity_repo, vector_index, embeddings)

    tagger = ConceptTagger(mention_extractor, linking_service)
    tagged = tagger.tag(text)

    assert tagged == []
    assert entity_repo.list_all() == []  # unmatched query term must not mint a new KG entity


def test_tag_blank_text_returns_empty(tmp_path: Path) -> None:
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    vector_index = FAISSVectorIndexRepository(dimension=_DIMENSION)
    mention_extractor = MentionExtractor(_FakeNLP([]), _EmptyKeyphraseExtractor(), _EmptyKeyphraseExtractor())
    linking_service = EntityLinkingService(entity_repo, vector_index, _FakeEmbeddingProvider({}))

    tagger = ConceptTagger(mention_extractor, linking_service)

    assert tagger.tag("   ") == []
