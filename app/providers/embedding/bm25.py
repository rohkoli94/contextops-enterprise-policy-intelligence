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

    The FastEmbed BM25 model is initialized once when the
    provider is created and reused for subsequent requests.

    The model name and cache location are configurable through
    application settings.

    Query-time sparse embedding supports an asynchronous path.
    """

    def __init__(self) -> None:
        # --------------------------------------------------
        # MODEL CACHE
        # --------------------------------------------------

        cache_dir = Path(
            settings.fastembed_cache_dir
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------

        self.model = SparseTextEmbedding(
            model_name=settings.bm25_model_name,
            cache_dir=str(cache_dir),
        )

    # ========================================================
    # SYNCHRONOUS SINGLE GENERATION
    # ========================================================

    def generate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a BM25 sparse representation for one text.
        """

        self._validate_text(text)

        embedding = next(
            self.model.embed([text])
        )

        return self._to_sparse_embedding(
            embedding
        )

    # ========================================================
    # ASYNCHRONOUS SINGLE GENERATION
    # ========================================================

    async def agenerate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a BM25 sparse representation asynchronously.

        FastEmbed inference is synchronous/local, so the blocking
        model operation is executed in a worker thread rather than
        blocking the FastAPI event loop.
        """

        self._validate_text(text)

        embedding = await asyncio.to_thread(
            self._generate_single_embedding,
            text,
        )

        return self._to_sparse_embedding(
            embedding
        )

    # ========================================================
    # SYNCHRONOUS BATCH GENERATION
    # ========================================================

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
            self._validate_text(text)

        embeddings = self.model.embed(texts)

        return [
            self._to_sparse_embedding(
                embedding
            )
            for embedding in embeddings
        ]

    # ========================================================
    # INTERNAL SINGLE EMBEDDING
    # ========================================================

    def _generate_single_embedding(
        self,
        text: str,
    ):
        """
        Run the blocking FastEmbed inference for one text.

        This method is intentionally synchronous because it is
        executed inside asyncio.to_thread().
        """

        return next(
            self.model.embed([text])
        )

    # ========================================================
    # SPARSE EMBEDDING MAPPING
    # ========================================================

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

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_text(
        text: str,
    ) -> None:
        """
        Validate sparse embedding input.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )