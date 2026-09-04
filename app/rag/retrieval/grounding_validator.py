from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class GroundingEvaluation:
    """
    Result of checking whether an answer is supported by
    the retrieved evidence.

    This is not a probability.
    It is a workflow decision.
    """

    grounded: bool
    confidence: str
    reason: str
    supported_sources: list[int]


class GroundingValidator(ABC):
    """
    Abstraction for validating generated answers against
    retrieved evidence.
    """

    @abstractmethod
    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        documents: list[Document],
    ) -> GroundingEvaluation:
        raise NotImplementedError