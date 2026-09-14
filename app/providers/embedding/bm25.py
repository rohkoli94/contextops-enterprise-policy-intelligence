import asyncio
from pathlib import Path

from fastembed import SparseTextEmbedding

from app.config.settings import settings
from app.domain.sparse_embedding import SparseEmbedding
from app.providers.embedding.sparse_base import (
    SparseEmbeddingProvider,
)


class BM25SparseEmbeddingProvider(
    SparseEmbeddingProvider
):
    """
    BM25 sparse embedding provider.

    FastEmbed inference is local and synchronous.

    Async methods therefore execute the blocking FastEmbed
    operation in a worker thread so the FastAPI event loop
    remains responsive.
    """

    def __init__(self) -> None:

        # ========================================================
        # CACHE DIRECTORY
        # ========================================================

        cache_dir = Path(
            settings.fastembed_cache_dir
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ========================================================
        # MODEL
        # ========================================================

        self.model = SparseTextEmbedding(
            model_name=settings.bm25_model_name,
            cache_dir=str(cache_dir),
        )

    # ============================================================
    # SYNCHRONOUS SINGLE GENERATION
    # ============================================================

    def generate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a BM25 sparse representation for one text.
        """

        self._validate_text(
            text
        )

        embedding = next(
            self.model.embed(
                [text]
            )
        )

        return self._to_sparse_embedding(
            embedding
        )

    # ============================================================
    # ASYNCHRONOUS SINGLE GENERATION
    # ============================================================

    async def agenerate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a BM25 sparse representation asynchronously.

        FastEmbed is synchronous/local, so inference is moved to
        a worker thread.
        """

        self._validate_text(
            text
        )

        embedding = await asyncio.to_thread(
            self._generate_single_embedding,
            text,
        )

        return self._to_sparse_embedding(
            embedding
        )

    # ============================================================
    # SYNCHRONOUS BATCH GENERATION
    # ============================================================

    def generate_batch(
        self,
        texts: list[str],
    ) -> list[SparseEmbedding]:
        """
        Generate BM25 sparse representations for multiple texts.
        """

        if not texts:
            return []

        for text in texts:
            self._validate_text(
                text
            )

        embeddings = self.model.embed(
            texts
        )

        return [
            self._to_sparse_embedding(
                embedding
            )
            for embedding in embeddings
        ]

    # ============================================================
    # ASYNCHRONOUS BATCH GENERATION
    # ============================================================

    async def agenerate_batch(
        self,
        texts: list[str],
    ) -> list[SparseEmbedding]:
        """
        Generate BM25 sparse representations asynchronously.

        FastEmbed's batch operation is synchronous/local, so the
        complete batch runs in one worker thread.

        This is more efficient than creating one worker task per
        chunk because FastEmbed already supports batch inference.
        """

        if texts is None:
            raise ValueError(
                "Sparse embedding texts cannot be None."
            )

        if not texts:
            return []

        for text in texts:
            self._validate_text(
                text
            )

        embeddings = await asyncio.to_thread(
            self._generate_batch_embeddings,
            texts,
        )

        return [
            self._to_sparse_embedding(
                embedding
            )
            for embedding in embeddings
        ]

    # ============================================================
    # INTERNAL SINGLE EMBEDDING
    # ============================================================

    def _generate_single_embedding(
        self,
        text: str,
    ):
        """
        Run blocking FastEmbed inference for one text.

        Called from asyncio.to_thread().
        """

        return next(
            self.model.embed(
                [text]
            )
        )

    # ============================================================
    # INTERNAL BATCH EMBEDDING
    # ============================================================

    def _generate_batch_embeddings(
        self,
        texts: list[str],
    ):
        """
        Run blocking FastEmbed batch inference.

        Called from asyncio.to_thread().
        """

        return list(
            self.model.embed(
                texts
            )
        )

    # ============================================================
    # SPARSE EMBEDDING MAPPING
    # ============================================================

    @staticmethod
    def _to_sparse_embedding(
        embedding,
    ) -> SparseEmbedding:
        """
        Convert FastEmbed output into the provider-neutral
        SparseEmbedding domain model.
        """

        return SparseEmbedding(
            indices=[
                int(index)
                for index in embedding.indices.tolist()
            ],
            values=[
                float(value)
                for value in embedding.values.tolist()
            ],
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_text(
        text: str,
    ) -> None:
        """
        Validate sparse embedding input.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Text must be a string."
            )

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )