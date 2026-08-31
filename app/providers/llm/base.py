from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    context: str | None = None


@dataclass
class VisionRequest:
    system_prompt: str
    user_prompt: str
    image_bytes: bytes
    media_type: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str


class LLMProvider(ABC):
    """
    Provider abstraction for text and vision LLM operations.

    The application can use either synchronous or asynchronous
    provider operations depending on the execution path.
    """

    # ========================================================
    # SYNCHRONOUS TEXT GENERATION
    # ========================================================

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a text response synchronously.
        """
        raise NotImplementedError

    # ========================================================
    # ASYNCHRONOUS TEXT GENERATION
    # ========================================================

    @abstractmethod
    async def agenerate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a text response asynchronously.
        """
        raise NotImplementedError

    # ========================================================
    # SYNCHRONOUS VISION GENERATION
    # ========================================================

    @abstractmethod
    def generate_vision(
        self,
        request: VisionRequest,
    ) -> LLMResponse:
        """
        Generate a response from a vision-capable LLM
        synchronously.
        """
        raise NotImplementedError

    # ========================================================
    # ASYNCHRONOUS VISION GENERATION
    # ========================================================

    @abstractmethod
    async def agenerate_vision(
        self,
        request: VisionRequest,
    ) -> LLMResponse:
        """
        Generate a response from a vision-capable LLM
        asynchronously.
        """
        raise NotImplementedError