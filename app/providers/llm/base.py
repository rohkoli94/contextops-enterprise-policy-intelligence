from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    context: str | None = None

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

