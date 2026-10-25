from pathlib import Path
from uuid import uuid4

from edukg.core.models import ConfidenceScore, Provenance, Triple
from edukg.core.models.enums import ValidationStatus
from edukg.modules.fusion.pending_triple_store import PendingTripleStore


def _triple() -> Triple:
    return Triple(
        subject_entity_id=uuid4(),
        predicate="is_related_to",
        object_entity_id=uuid4(),
        confidence=ConfidenceScore(value=0.3, method="test"),
        provenance=Provenance(source_document_id=uuid4(), extractor_name="test", extractor_version="1.0.0"),
        source_document_id=uuid4(),
        extractor_version="1.0.0",
        relation_id=uuid4(),
        validation_status=ValidationStatus.VALID,
    )


def test_get_returns_saved_triple(tmp_path: Path) -> None:
    store = PendingTripleStore(base_dir=tmp_path)
    triple = _triple()
    store.save(triple)

    assert store.get(triple.id).id == triple.id


def test_get_missing_returns_none(tmp_path: Path) -> None:
    assert PendingTripleStore(base_dir=tmp_path).get(uuid4()) is None


def test_remove_deletes_from_pending_list(tmp_path: Path) -> None:
    store = PendingTripleStore(base_dir=tmp_path)
    triple = _triple()
    store.save(triple)

    store.remove(triple.id)

    assert store.get(triple.id) is None
    assert store.list_pending() == []


def test_remove_nonexistent_is_a_noop(tmp_path: Path) -> None:
    store = PendingTripleStore(base_dir=tmp_path)
    store.remove(uuid4())  # must not raise
