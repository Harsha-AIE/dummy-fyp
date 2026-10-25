from pathlib import Path

import pytest

from edukg.core.exceptions import PipelineStageError
from edukg.core.models import Document, DocumentType
from edukg.core.models.enums import PipelineStage
from edukg.pipeline import CheckpointStore, PipelineContext, PipelineOrchestrator
from edukg.pipeline.orchestrator import STAGE_ORDER


def _make_context() -> PipelineContext:
    doc = Document(file_path=Path("data/raw/sample.pdf"), file_hash="c" * 64, document_type=DocumentType.PDF)
    return PipelineContext(document=doc)


def _noop_stage(stage: PipelineStage):
    def handler(context: PipelineContext) -> PipelineContext:
        context.mark_complete(stage)
        return context

    return handler


def test_on_stage_event_fires_started_then_completed_for_every_stage(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=tmp_path))
    for stage in STAGE_ORDER:
        orchestrator.register(stage, _noop_stage(stage))

    events: list[tuple[PipelineStage, str]] = []
    orchestrator.run(_make_context(), on_stage_event=lambda stage, status: events.append((stage, status)))

    assert events == [pair for stage in STAGE_ORDER for pair in ((stage, "started"), (stage, "completed"))]


def test_on_stage_event_fires_failed_on_stage_error(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=tmp_path))

    def failing_ingestion(context: PipelineContext) -> PipelineContext:
        raise PipelineStageError("ingestion", "simulated failure")

    orchestrator.register(PipelineStage.INGESTION, failing_ingestion)
    for stage in STAGE_ORDER[1:]:
        orchestrator.register(stage, _noop_stage(stage))

    events: list[tuple[PipelineStage, str]] = []
    with pytest.raises(PipelineStageError):
        orchestrator.run(_make_context(), on_stage_event=lambda stage, status: events.append((stage, status)))

    assert events == [(PipelineStage.INGESTION, "started"), (PipelineStage.INGESTION, "failed")]


def test_run_without_callback_is_unaffected(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=tmp_path))
    for stage in STAGE_ORDER:
        orchestrator.register(stage, _noop_stage(stage))

    result = orchestrator.run(_make_context())

    assert result.completed_stages == list(STAGE_ORDER)
