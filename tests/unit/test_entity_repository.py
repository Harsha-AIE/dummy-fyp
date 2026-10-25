from pathlib import Path

from edukg.core.models import Entity
from edukg.core.models.enums import EntityType
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository


def test_save_and_find_entity_with_newline_in_name_does_not_crash(tmp_path: Path) -> None:
    """Regression test: a KeyBERT/YAKE keyphrase can span a paragraph break and
    include a literal newline (e.g. "Novel Antibiotics\\nGenome"), which is an
    invalid Windows filename character if not sanitized out of the name-index
    key — this previously raised CacheError and aborted the pipeline."""
    repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    entity = Entity(canonical_name="Novel Antibiotics\nGenome", entity_type=EntityType.CONCEPT)

    saved = repo.save(entity)

    assert repo.get(saved.id) is not None
    assert repo.find_by_canonical_name("Novel Antibiotics\nGenome") is not None
    assert repo.find_by_canonical_name("Novel Antibiotics\nGenome").id == saved.id


def test_save_entity_with_other_unsafe_characters(tmp_path: Path) -> None:
    repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")
    entity = Entity(canonical_name='Question: "What/is?this*" <thing>|here', entity_type=EntityType.CONCEPT)

    repo.save(entity)

    assert repo.find_by_canonical_name('Question: "What/is?this*" <thing>|here') is not None
