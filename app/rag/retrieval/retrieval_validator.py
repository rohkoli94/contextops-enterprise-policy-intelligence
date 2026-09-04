from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalEvaluation:
    """
    Result of retrieval-quality evaluation.

    This is intentionally not a probability.

    `confidence` is a qualitative classification based on the
    available retrieval signals.

    Day 19 can introduce a richer calibrated scoring strategy.
    """

    sufficient: bool
    confidence: str
    score: float | None
    reason: str
    signals: dict[str, object]


class RetrievalValidator(ABC):
    """
    Application-level abstraction for evaluating whether the
    retrieved evidence is sufficient to continue toward answer
    generation.
    """

    @abstractmethod
    async def evaluate(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> RetrievalEvaluation:
        raise NotImplementedError