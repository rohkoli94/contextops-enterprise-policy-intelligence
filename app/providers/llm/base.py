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
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Genarate response from LLM"""
        raise NotImplementedError

    @abstractmethod
    def generate_vision(
        self,
        request: VisionRequest,
    ) -> LLMResponse:
        """Generate a response from a vision-capable LLM."""
        raise NotImplementedError