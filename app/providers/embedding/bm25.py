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

    def generate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a BM25 sparse representation for one text.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        embedding = next(
            self.model.embed([text])
        )

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
            if not text or not text.strip():
                raise ValueError(
                    "Texts cannot contain empty values."
                )

        embeddings = self.model.embed(texts)

        return [
            SparseEmbedding(
                indices=[
                    int(index)
                    for index in embedding.indices.tolist()
                ],
                values=[
                    float(value)
                    for value in embedding.values.tolist()
                ],
            )
            for embedding in embeddings
        ]