from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingRequest:
    text: str


@dataclass
class EmbeddingBatchRequest:
    texts: list[str]


@dataclass
class EmbeddingResponse:
    vector: list[float]
    model: str
    provider: str


@dataclass
class EmbeddingBatchResponse:
    vectors: list[list[float]]
    model: str
    provider: str


class EmbeddingProvider(ABC):

    @abstractmethod
    def generate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate an embedding for a single text."""
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by this
        embedding provider.
        """
        raise NotImplementedError