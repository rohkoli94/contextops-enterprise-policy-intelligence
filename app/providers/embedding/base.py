import asyncio
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

    # ============================================================
    # SYNCHRONOUS SINGLE EMBEDDING
    # ============================================================

    @abstractmethod
    def generate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """
        Generate an embedding for a single text.
        """
        raise NotImplementedError

    # ============================================================
    # ASYNCHRONOUS SINGLE EMBEDDING
    # ============================================================

    @abstractmethod
    async def agenerate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """
        Generate an embedding asynchronously for a single text.
        """
        raise NotImplementedError

    # ============================================================
    # SYNCHRONOUS BATCH EMBEDDING
    # ============================================================

    @abstractmethod
    def generate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        """
        Generate embeddings for multiple texts.
        """
        raise NotImplementedError

    # ============================================================
    # ASYNCHRONOUS BATCH EMBEDDING
    # ============================================================

    async def agenerate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        """
        Generate embeddings for multiple texts asynchronously.

        Default implementation:

        - preserves input order
        - uses the provider's async single-item method
        - bounds concurrency
        - keeps provider-specific batching decisions outside
          the orchestration layer

        Concrete providers with native async batch APIs should
        override this method.
        """

        if request is None:
            raise ValueError(
                "Embedding batch request cannot be None."
            )

        if not request.texts:
            raise ValueError(
                "Embedding batch request cannot be empty."
            )

        concurrency_limit = 8

        semaphore = asyncio.Semaphore(
            concurrency_limit
        )

        async def generate_one(
            index: int,
            text: str,
        ) -> tuple[int, EmbeddingResponse]:

            if not isinstance(text, str):
                raise TypeError(
                    "Embedding input text must be a string."
                )

            if not text.strip():
                raise ValueError(
                    "Embedding input text cannot be empty."
                )

            async with semaphore:

                response = await self.agenerate(
                    EmbeddingRequest(
                        text=text,
                    )
                )

                return index, response

        results = await asyncio.gather(
            *(
                generate_one(
                    index=index,
                    text=text,
                )
                for index, text in enumerate(
                    request.texts
                )
            )
        )

        # Explicitly preserve input ordering.
        results.sort(
            key=lambda item: item[0]
        )

        responses = [
            response
            for _, response in results
        ]

        first_response = responses[0]

        vectors = [
            response.vector
            for response in responses
        ]

        # All embeddings within one batch should come from
        # the same provider and logical model.
        for response in responses:

            if response.model != first_response.model:
                raise ValueError(
                    "Embedding model changed within "
                    "a single batch response."
                )

            if response.provider != first_response.provider:
                raise ValueError(
                    "Embedding provider changed within "
                    "a single batch response."
                )

        return EmbeddingBatchResponse(
            vectors=vectors,
            model=first_response.model,
            provider=first_response.provider,
        )

    # ============================================================
    # VECTOR DIMENSION
    # ============================================================

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by this
        embedding provider.
        """
        raise NotImplementedError