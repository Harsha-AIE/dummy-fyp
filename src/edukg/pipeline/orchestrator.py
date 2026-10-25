from __future__ import annotations

from collections.abc import Callable

from edukg.core.exceptions import PipelineStageError
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.pipeline.checkpoint import CheckpointStore
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)

# Mandatory, fixed execution order. No stage may be skipped or reordered.
STAGE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.INGESTION,
    PipelineStage.SUBJECT_DETECTION,
    PipelineStage.ONTOLOGY_LOAD,
    PipelineStage.ONTOLOGY_EXTENSION,
    PipelineStage.ONTOLOGY_VALIDATION,
    PipelineStage.ENTITY_EXTRACTION,
    PipelineStage.ENTITY_LINKING,
    PipelineStage.RELATION_EXTRACTION,
    PipelineStage.TRIPLE_GENERATION,
    PipelineStage.TRIPLE_VALIDATION,
    PipelineStage.KNOWLEDGE_FUSION,
    PipelineStage.GRAPH_STORAGE,
)

StageFn = Callable[[PipelineContext], PipelineContext]
StageEventFn = Callable[[PipelineStage, str], None]


class PipelineOrchestrator:
    """
    Drives a PipelineContext through STAGE_ORDER using a registry of stage
    callables. Stages are injected (dependency inversion) — the orchestrator
    has zero knowledge of Neo4j, OpenAI, PaddleOCR, or FAISS.

    A stage callable must accept and return a PipelineContext, and raise a
    PipelineStageError subclass on failure. On success it should call
    `context.mark_complete(stage)` itself before returning.
    """

    def __init__(self, checkpoint_store: CheckpointStore | None = None) -> None:
        self._registry: dict[PipelineStage, StageFn] = {}
        self._checkpoints = checkpoint_store or CheckpointStore()

    def register(self, stage: PipelineStage, handler: StageFn) -> "PipelineOrchestrator":
        self._registry[stage] = handler
        return self

    def _validate_registry(self, up_to: PipelineStage | None) -> None:
        target_index = len(STAGE_ORDER) - 1 if up_to is None else STAGE_ORDER.index(up_to)
        missing = [s for s in STAGE_ORDER[: target_index + 1] if s not in self._registry]
        if missing:
            raise PipelineStageError(
                "orchestrator",
                f"No handler registered for required stage(s): {[s.value for s in missing]}",
            )

    def run(
        self,
        context: PipelineContext,
        *,
        stop_after: PipelineStage | None = None,
        checkpoint_after_each_stage: bool = True,
        on_stage_event: StageEventFn | None = None,
    ) -> PipelineContext:
        """on_stage_event, if given, is called as (stage, status) at the same
        points the structured logger already fires — status is one of
        "started" / "completed" / "failed" / "skipped". Purely additive: the
        API layer's pipeline dashboard uses this to stream live progress;
        every existing caller passes nothing and is unaffected."""
        self._validate_registry(stop_after)
        target_index = len(STAGE_ORDER) - 1 if stop_after is None else STAGE_ORDER.index(stop_after)

        for stage in STAGE_ORDER[: target_index + 1]:
            if context.is_complete_through(stage):
                logger.info("stage_skipped_already_complete", stage=stage.value, document_id=str(context.document.id))
                if on_stage_event is not None:
                    on_stage_event(stage, "skipped")
                continue

            handler = self._registry[stage]
            logger.info("stage_started", stage=stage.value, document_id=str(context.document.id))
            if on_stage_event is not None:
                on_stage_event(stage, "started")
            try:
                context = handler(context)
            except PipelineStageError as exc:
                context.stage_errors[stage.value] = exc.message
                logger.error("stage_failed", stage=stage.value, error=exc.message, document_id=str(context.document.id))
                if on_stage_event is not None:
                    on_stage_event(stage, "failed")
                if checkpoint_after_each_stage:
                    self._checkpoints.save(context)
                raise
            except Exception as exc:  # noqa: BLE001 - convert unexpected errors into typed pipeline errors
                wrapped = PipelineStageError(stage.value, f"Unexpected error: {exc}")
                context.stage_errors[stage.value] = wrapped.message
                logger.error("stage_failed_unexpected", stage=stage.value, error=str(exc))
                if on_stage_event is not None:
                    on_stage_event(stage, "failed")
                if checkpoint_after_each_stage:
                    self._checkpoints.save(context)
                raise wrapped from exc

            context.mark_complete(stage)
            logger.info("stage_completed", stage=stage.value, document_id=str(context.document.id))
            if on_stage_event is not None:
                on_stage_event(stage, "completed")

            if checkpoint_after_each_stage:
                self._checkpoints.save(context)

        return context

    def checkpoint_store(self) -> CheckpointStore:
        return self._checkpoints
