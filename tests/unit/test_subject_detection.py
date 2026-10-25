from pathlib import Path

import pytest

from edukg.core.exceptions import SubjectDetectionError
from edukg.core.models import Document, DocumentType, Subject
from edukg.core.models.enums import ExtractionMethod
from edukg.modules.subject_detection.keyword_strategy import KeywordSubjectDetectionStrategy
from edukg.modules.subject_detection.subject_detection_service import SubjectDetectionStageHandler
from edukg.pipeline.context import PipelineContext
from edukg.core.models import Chunk
from uuid import uuid4


def test_keyword_strategy_detects_machine_learning() -> None:
    strategy = KeywordSubjectDetectionStrategy()
    text = "We use gradient descent to minimize the loss function during training on the training set."
    subject, confidence = strategy.detect(text)
    assert subject == Subject.MACHINE_LEARNING
    assert confidence > 0.0


def test_keyword_strategy_returns_unknown_for_no_signal() -> None:
    strategy = KeywordSubjectDetectionStrategy()
    subject, confidence = strategy.detect("The quick brown fox jumps over the lazy dog.")
    assert subject == Subject.UNKNOWN
    assert confidence == 0.0


def test_keyword_strategy_handles_empty_text() -> None:
    strategy = KeywordSubjectDetectionStrategy()
    assert strategy.detect("") == (Subject.UNKNOWN, 0.0)


def _make_context(text: str) -> PipelineContext:
    doc = Document(file_path=Path("data/raw/sample.pdf"), file_hash="c" * 64, document_type=DocumentType.PDF)
    chunk = Chunk(
        document_id=doc.id,
        sequence_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        token_count=max(1, len(text) // 4),
    )
    context = PipelineContext(document=doc)
    context.chunks = [chunk]
    return context


def test_subject_detection_stage_handler_updates_document() -> None:
    handler = SubjectDetectionStageHandler(KeywordSubjectDetectionStrategy())
    context = _make_context("Backpropagation and gradient descent are core to neural network training.")

    result = handler.run(context)

    assert result.document.detected_subject == Subject.MACHINE_LEARNING
    assert result.document.subject_confidence is not None
    assert result.is_complete_through(handler.stage)


def test_subject_detection_stage_handler_requires_chunks() -> None:
    handler = SubjectDetectionStageHandler(KeywordSubjectDetectionStrategy())
    doc = Document(file_path=Path("data/raw/sample.pdf"), file_hash="d" * 64, document_type=DocumentType.PDF)
    context = PipelineContext(document=doc)

    with pytest.raises(SubjectDetectionError):
        handler.run(context)
