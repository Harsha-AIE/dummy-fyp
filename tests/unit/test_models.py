from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from edukg.core.models import (
    Chunk,
    ConfidenceScore,
    Document,
    DocumentType,
    Entity,
    EntityType,
    ExtractionMethod,
    Provenance,
    Relation,
    RelationDirection,
    Subject,
    Triple,
)


def _make_document() -> Document:
    return Document(
        file_path=Path("data/raw/sample.pdf"),
        file_hash="a" * 64,
        document_type=DocumentType.PDF,
    )


def test_document_defaults() -> None:
    doc = _make_document()
    assert doc.detected_subject == Subject.UNKNOWN
    assert doc.metadata.language == "en"
    assert doc.id is not None


def test_chunk_requires_non_empty_text() -> None:
    doc = _make_document()
    with pytest.raises(ValidationError):
        Chunk(
            document_id=doc.id,
            sequence_index=0,
            text="",
            char_start=0,
            char_end=0,
            extraction_method=ExtractionMethod.NATIVE_TEXT,
            token_count=0,
        )


def test_confidence_score_bounds() -> None:
    with pytest.raises(ValidationError):
        ConfidenceScore(value=1.5, method="cosine_similarity")

    score = ConfidenceScore(value=0.9, method="cosine_similarity")
    assert score.meets_threshold(0.85)
    assert not score.meets_threshold(0.95)


def test_relation_rejects_self_loop() -> None:
    entity_id = uuid4()
    prov = Provenance(
        source_document_id=uuid4(),
        extractor_name="relation_extractor",
        extractor_version="0.1.0",
    )
    with pytest.raises(ValidationError):
        Relation(
            subject_entity_id=entity_id,
            predicate="is_prerequisite_of",
            object_entity_id=entity_id,
            direction=RelationDirection.DIRECTED,
            confidence=ConfidenceScore(value=0.8, method="rule_based"),
            provenance=prov,
            ontology_relation_id=uuid4(),
        )


def test_entity_alias_deduplication() -> None:
    entity = Entity(canonical_name="Gradient Descent", entity_type=EntityType.ALGORITHM)
    entity.add_alias("gradient descent")  # case-insensitive duplicate of canonical name
    entity.add_alias("GD")
    entity.add_alias("GD")  # duplicate alias
    assert entity.aliases == ["GD"]


def test_triple_natural_key_stable() -> None:
    doc_id = uuid4()
    subj_id = uuid4()
    obj_id = uuid4()
    prov = Provenance(source_document_id=doc_id, extractor_name="triple_gen", extractor_version="0.1.0")
    triple = Triple(
        subject_entity_id=subj_id,
        predicate="is_prerequisite_of",
        object_entity_id=obj_id,
        confidence=ConfidenceScore(value=0.7, method="rule_based"),
        provenance=prov,
        source_document_id=doc_id,
        extractor_version="0.1.0",
        relation_id=uuid4(),
    )
    assert triple.natural_key() == (subj_id, "is_prerequisite_of", obj_id)
