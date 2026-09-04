from abc import ABC, abstractmethod

from app.guardrails.models import GuardrailResult, PIIAnalysisResult


class Guardrail(ABC):
    """
    Base contract for asynchronous guardrails.

    Guardrails are intentionally provider-independent.
    The implementation can later use rules, a classifier,
    an external DLP service, or an LLM-backed detector.
    """

    @abstractmethod
    async def evaluate(self, **kwargs: object) -> GuardrailResult:
        raise NotImplementedError


class PIIAnalyzer(ABC):
    """
    Contract for detecting personally identifiable information.

    The same analyzer can be used for:
    - user input
    - retrieved context
    - generated output
    """

    @abstractmethod
    async def analyze(self, text: str) -> PIIAnalysisResult:
        raise NotImplementedError