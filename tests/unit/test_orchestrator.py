from pathlib import Path

import pytest

from edukg.core.exceptions import PipelineStageError
from edukg.core.models import Document, DocumentType
from edukg.core.models.enums import PipelineStage
from edukg.pipeline import CheckpointStore, PipelineContext, PipelineOrchestrator
from edukg.pipeline.orchestrator import STAGE_ORDER


def _make_context() -> PipelineContext:
    doc = Document(file_path=Path("data/raw/sample.pdf"), file_hash="b" * 64, document_type=DocumentType.PDF)
    return PipelineContext(document=doc)


def _noop_stage(stage: PipelineStage):
    def handler(context: PipelineContext) -> PipelineContext:
        context.mark_complete(stage)
        return context

    return handler


def test_orchestrator_rejects_missing_handlers() -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=Path("/tmp/edukg_test_ckpt")))
    with pytest.raises(PipelineStageError):
        orchestrator.run(_make_context())


def test_orchestrator_runs_all_stages_in_order(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=tmp_path))
    for stage in STAGE_ORDER:
        orchestrator.register(stage, _noop_stage(stage))

    result = orchestrator.run(_make_context())
    assert result.completed_stages == list(STAGE_ORDER)


def test_orchestrator_stops_after_requested_stage(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=tmp_path))
    for stage in STAGE_ORDER:
        orchestrator.register(stage, _noop_stage(stage))

    result = orchestrator.run(_make_context(), stop_after=PipelineStage.SUBJECT_DETECTION)
    assert result.completed_stages == [PipelineStage.INGESTION, PipelineStage.SUBJECT_DETECTION]


def test_orchestrator_propagates_stage_failure(tmp_path) -> None:
    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore(base_dir=tmp_path))

    def failing_ingestion(context: PipelineContext) -> PipelineContext:
        raise PipelineStageError("ingestion", "simulated failure")

    orchestrator.register(PipelineStage.INGESTION, failing_ingestion)
    for stage in STAGE_ORDER[1:]:
        orchestrator.register(stage, _noop_stage(stage))

    context = _make_context()
    with pytest.raises(PipelineStageError):
        orchestrator.run(context)
    assert "ingestion" in context.stage_errors


def test_checkpoint_round_trip(tmp_path) -> None:
    store = CheckpointStore(base_dir=tmp_path)
    context = _make_context()
    context.mark_complete(PipelineStage.INGESTION)

    store.save(context)
    restored = store.load(context.document.id)

    assert restored is not None
    assert restored.document.id == context.document.id
    assert restored.completed_stages == [PipelineStage.INGESTION]


def test_checkpoint_export_path_available_for_download(tmp_path) -> None:
    store = CheckpointStore(base_dir=tmp_path)
    context = _make_context()
    store.save(context)

    exported = store.export_path(context.document.id)
    assert exported is not None
    assert exported.exists()
