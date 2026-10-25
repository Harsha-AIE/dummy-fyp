"""DKTInferenceService's model-loading logic — real torch round trips
against a tmp_path model dir, not mocked, since the thing under test is
specifically "does the saved file format round-trip into a working model."
"""

import json
from uuid import uuid4

import torch

from edukg.agents.knowledge_tracing.dkt_model import DKTModel
from edukg.agents.knowledge_tracing.dkt_service import DKTInferenceService
from edukg.core.models.enums import KTEventSource
from edukg.core.models.knowledge_tracing import KTEvent


def _save_model(model_dir, vocab: dict[str, int], d_model: int, *, legacy_format: bool = False) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = DKTModel(vocab_size=len(vocab), d_model=d_model)
    torch.save(model.state_dict(), model_dir / "dkt_model.pt")
    payload = vocab if legacy_format else {"vocab": vocab, "d_model": d_model}
    (model_dir / "dkt_vocab.json").write_text(json.dumps(payload), encoding="utf-8")


def test_loads_new_format_with_explicit_d_model(tmp_path) -> None:
    concept = uuid4()
    vocab = {str(concept): 0}
    _save_model(tmp_path, vocab, d_model=64)

    service = DKTInferenceService(model_dir=tmp_path)

    assert service.is_available is True


def test_loads_legacy_flat_vocab_format_defaulting_d_model_to_32(tmp_path) -> None:
    concept = uuid4()
    vocab = {str(concept): 0}
    _save_model(tmp_path, vocab, d_model=32, legacy_format=True)

    service = DKTInferenceService(model_dir=tmp_path)

    assert service.is_available is True


def test_mismatched_d_model_fails_to_load_rather_than_silently_misreading_weights(tmp_path) -> None:
    # The model file was saved with d_model=64, but the vocab.json claims 32
    # — a corrupted/hand-edited pairing. load_state_dict() must reject this
    # (tensor shape mismatch) rather than loading garbage into a
    # wrong-shaped model and serving nonsense predictions silently.
    concept = uuid4()
    vocab = {str(concept): 0}
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = DKTModel(vocab_size=len(vocab), d_model=64)
    torch.save(model.state_dict(), tmp_path / "dkt_model.pt")
    (tmp_path / "dkt_vocab.json").write_text(json.dumps({"vocab": vocab, "d_model": 32}), encoding="utf-8")

    service = DKTInferenceService(model_dir=tmp_path)

    assert service.is_available is False


def test_predict_works_end_to_end_after_loading_new_format(tmp_path) -> None:
    concept_a = uuid4()
    concept_b = uuid4()
    vocab = {str(concept_a): 0, str(concept_b): 1}
    _save_model(tmp_path, vocab, d_model=32)
    service = DKTInferenceService(model_dir=tmp_path)
    assert service.is_available

    events = [
        KTEvent(student_id="s1", concept_id=concept_a, source=KTEventSource.DIALOGUE, correct=True),
        KTEvent(student_id="s1", concept_id=concept_b, source=KTEventSource.DIALOGUE, correct=False),
    ]

    prediction = service.predict(concept_a, events)

    assert prediction is not None
    assert 0.0 <= prediction.mastery <= 1.0


def test_predict_returns_none_for_a_concept_outside_the_trained_vocab(tmp_path) -> None:
    known_concept = uuid4()
    unknown_concept = uuid4()
    vocab = {str(known_concept): 0}
    _save_model(tmp_path, vocab, d_model=32)
    service = DKTInferenceService(model_dir=tmp_path)

    events = [KTEvent(student_id="s1", concept_id=known_concept, source=KTEventSource.DIALOGUE, correct=True)]

    assert service.predict(unknown_concept, events) is None
