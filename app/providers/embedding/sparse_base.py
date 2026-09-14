import asyncio
from abc import ABC, abstractmethod

from app.domain.sparse_embedding import SparseEmbedding


class SparseEmbeddingProvider(ABC):
    """
    Abstraction for sparse embedding generation.

    Supports:

        - synchronous single generation
        - asynchronous single generation
        - synchronous batch generation
        - asynchronous batch generation

    The application remains provider-agnostic. Concrete
    implementations decide whether async execution is native
    or delegated to worker threads.
    """

    # ============================================================
    # SYNCHRONOUS SINGLE GENERATION
    # ============================================================

    @abstractmethod
    def generate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a sparse representation for one text.
        """
        raise NotImplementedError

    # ============================================================
    # ASYNCHRONOUS SINGLE GENERATION
    # ============================================================

    @abstractmethod
    async def agenerate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a sparse representation asynchronously
        for one text.
        """
        raise NotImplementedError

    # ============================================================
    # SYNCHRONOUS BATCH GENERATION
    # ============================================================

    @abstractmethod
    def generate_batch(
        self,
        texts: list[str],
    ) -> list[SparseEmbedding]:
        """
        Generate sparse representations for multiple texts.
        """
        raise NotImplementedError

    # ============================================================
    # ASYNCHRONOUS BATCH GENERATION
    # ============================================================

    async def agenerate_batch(
        self,
        texts: list[str],
    ) -> list[SparseEmbedding]:
        """
        Generate sparse representations asynchronously.

        Default implementation uses the provider's async single
        generation method with bounded concurrency.

        Concrete providers that have a more efficient native
        asynchronous batch operation can override this method.
        """

        if texts is None:
            raise ValueError(
                "Sparse embedding texts cannot be None."
            )

        if not texts:
            return []

        concurrency_limit = 8

        semaphore = asyncio.Semaphore(
            concurrency_limit
        )

        async def generate_one(
            index: int,
            text: str,
        ) -> tuple[int, SparseEmbedding]:

            if not isinstance(text, str):
                raise TypeError(
                    "Sparse embedding input text "
                    "must be a string."
                )

            if not text.strip():
                raise ValueError(
                    "Sparse embedding input text "
                    "cannot be empty."
                )

            async with semaphore:
                embedding = await self.agenerate(
                    text
                )

                return index, embedding

        results = await asyncio.gather(
            *(
                generate_one(
                    index=index,
                    text=text,
                )
                for index, text in enumerate(
                    texts
                )
            )
        )

        # Explicitly restore original input ordering.
        results.sort(
            key=lambda item: item[0]
        )

        return [
            embedding
            for _, embedding in results
        ]