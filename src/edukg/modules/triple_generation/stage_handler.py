from __future__ import annotations

from edukg.core.exceptions import TripleGenerationError
from edukg.core.interfaces import PipelineStageHandler
from edukg.core.logging_setup import get_logger
from edukg.core.models.enums import PipelineStage
from edukg.modules.triple_generation.triple_generator import TripleGenerator
from edukg.pipeline.context import PipelineContext

logger = get_logger(__name__)


class TripleGenerationStageHandler(PipelineStageHandler[PipelineContext, PipelineContext]):
    stage = PipelineStage.TRIPLE_GENERATION

    def __init__(self, generator: TripleGenerator | None = None) -> None:
        self._generator = generator or TripleGenerator()

    def run(self, input_data: PipelineContext) -> PipelineContext:
        context = input_data
        if not context.relations:
            raise TripleGenerationError("No relations available — relation extraction must run before triple generation")

        triples = self._generator.generate(context.relations)

        context.triples = triples
        context.mark_complete(self.stage)
        logger.info("stage_triple_generation_done", document_id=str(context.document.id), triple_count=len(triples))
        return context
