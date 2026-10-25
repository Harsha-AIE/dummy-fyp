"""Run: python scripts/verify_bootstrap.py
Confirms Module 1 (config, logging, models, interfaces, orchestrator) is wired correctly."""

from __future__ import annotations

from pathlib import Path

from edukg.core.config import get_settings
from edukg.core.logging_setup import configure_logging, get_logger
from edukg.core.models import Document, DocumentType
from edukg.core.models.enums import PipelineStage
from edukg.pipeline import CheckpointStore, PipelineContext, PipelineOrchestrator
from edukg.pipeline.orchestrator import STAGE_ORDER

configure_logging()
logger = get_logger("verify_bootstrap")


def _noop(stage: PipelineStage):
    def handler(context: PipelineContext) -> PipelineContext:
        context.mark_complete(stage)
        return context

    return handler


def main() -> None:
    settings = get_settings()
    logger.info("settings_loaded", environment=settings.environment, data_dir=str(settings.paths.data_dir))

    document = Document(file_path=Path("data/raw/demo.pdf"), file_hash="0" * 64, document_type=DocumentType.PDF)

    orchestrator = PipelineOrchestrator(checkpoint_store=CheckpointStore())
    for stage in STAGE_ORDER:
        orchestrator.register(stage, _noop(stage))

    result = orchestrator.run(PipelineContext(document=document))
    logger.info("pipeline_complete", stages=[s.value for s in result.completed_stages])

    checkpoint_path = orchestrator.checkpoint_store().export_path(document.id)
    logger.info("checkpoint_available_for_download", path=str(checkpoint_path))


if __name__ == "__main__":
    main()
