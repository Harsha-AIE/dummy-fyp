from __future__ import annotations

from edukg.core.exceptions import SubjectDetectionError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.subject_detection.strategy import SubjectDetectionStrategy
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class SubjectDetectionStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    """
    Automatically determines the document's domain from its chunk text and writes it
    onto context.document. The user never selects an ontology manually — this stage
    is the sole source of truth for `document.detected_subject`.
    """

    stage = PipelineStage.SUBJECT_DETECTION

    def __init__(self, strategy: SubjectDetectionStrategy) -> None:
        self._strategy = strategy

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.chunks:
            raise SubjectDetectionError("No chunks available — ingestion must run before subject detection")

        combined_text = "\n".join(chunk.text for chunk in context.chunks)
        subject, confidence = self._strategy.detect(combined_text)

        context.document.detected_subject = subject
        context.document.subject_confidence = confidence
        context.document.touch()
        context.mark_complete(self.stage)

        logger.info(
            "stage_subject_detection_done",
            document_id=str(context.document.id),
            subject=subject.value,
            confidence=confidence,
            strategy=self._strategy.name(),
        )
        return context
