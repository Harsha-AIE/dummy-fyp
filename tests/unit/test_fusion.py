from pathlib import Path
from uuid import uuid4

import pytest

from edukg.core.models import ConfidenceScore, Provenance, Triple
from edukg.core.models.enums import ValidationStatus
from edukg.modules.fusion.fusion_service import KnowledgeFusionService
from edukg.modules.fusion.pending_triple_store import PendingTripleStore


def _triple(subject_id, object_id, predicate: str, confidence: float, method: str, status: ValidationStatus) -> Triple:
    return Triple(
        subject_entity_id=subject_id,
        predicate=predicate,
        object_entity_id=object_id,
        confidence=ConfidenceScore(value=confidence, method=method),
        provenance=Provenance(source_document_id=uuid4(), extractor_name="test", extractor_version="1.0.0"),
        source_document_id=uuid4(),
        extractor_version="1.0.0",
        relation_id=uuid4(),
        validation_status=status,
    )


def test_duplicate_triples_fuse_with_agreement_and_evidence_bonus() -> None:
    subject_id, object_id = uuid4(), uuid4()
    triples = [
        _triple(subject_id, object_id, "is_related_to", 0.6, "cue_phrase", ValidationStatus.VALID),
        _triple(subject_id, object_id, "is_related_to", 0.5, "dependency_parse", ValidationStatus.VALID),
        _triple(subject_id, object_id, "is_related_to", 0.55, "co_occurrence", ValidationStatus.WARNING),
    ]

    accepted, pending = KnowledgeFusionService(min_confidence=0.5).fuse(triples)

    assert len(accepted) == 1
    assert pending == []
    fused = accepted[0]
    # base (0.6) + agreement bonus (0.10, >1 distinct method) + evidence bonus (2 extra * 0.05 = 0.10)
    assert fused.confidence.value == pytest.approx(0.8)
    assert fused.confidence.method == "fusion"
    assert fused.validation_status == ValidationStatus.WARNING  # worst status across the group


def test_low_confidence_triple_routes_to_pending() -> None:
    triple = _triple(uuid4(), uuid4(), "is_related_to", 0.2, "co_occurrence", ValidationStatus.VALID)

    accepted, pending = KnowledgeFusionService(min_confidence=0.5).fuse([triple])

    assert accepted == []
    assert len(pending) == 1


def test_invalid_triple_routes_to_pending_even_with_high_confidence() -> None:
    triple = _triple(uuid4(), uuid4(), "is_related_to", 0.95, "cue_phrase", ValidationStatus.INVALID)

    accepted, pending = KnowledgeFusionService(min_confidence=0.5).fuse([triple])

    assert accepted == []
    assert len(pending) == 1


def test_pending_triple_store_round_trip(tmp_path: Path) -> None:
    store = PendingTripleStore(base_dir=tmp_path / "pending_triples")
    triple = _triple(uuid4(), uuid4(), "is_related_to", 0.2, "co_occurrence", ValidationStatus.VALID)

    store.save(triple)
    reloaded = store.list_pending()

    assert len(reloaded) == 1
    assert reloaded[0].id == triple.id
