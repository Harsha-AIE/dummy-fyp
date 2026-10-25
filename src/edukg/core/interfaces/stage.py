from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from edukg.core.models.enums import PipelineStage

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class PipelineStageHandler(ABC, Generic[TIn, TOut]):
    """
    Base contract every pipeline stage implementation must satisfy.

    Each stage is independently testable and replaceable: it consumes a
    well-defined input type and produces a well-defined output type, with
    no knowledge of neighboring stages.
    """

    stage: PipelineStage

    @abstractmethod
    def run(self, input_data: TIn) -> TOut:
        """Execute the stage. Must raise a PipelineStageError subclass on failure."""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
