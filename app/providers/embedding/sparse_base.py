from abc import ABC, abstractmethod

from app.domain.sparse_embedding import SparseEmbedding


class SparseEmbeddingProvider(ABC):
    """
    Abstraction for sparse embedding generation.

    Implementations can generate lexical/sparse
    representations such as BM25.

    The query/retrieval path supports asynchronous
    sparse embedding generation.
    """

    # ========================================================
    # SYNCHRONOUS SINGLE GENERATION
    # ========================================================

    @abstractmethod
    def generate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a sparse representation for one text.
        """
        raise NotImplementedError

    # ========================================================
    # ASYNCHRONOUS SINGLE GENERATION
    # ========================================================

    @abstractmethod
    async def agenerate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a sparse representation asynchronously
        for one text.

        Used by the asynchronous query/retrieval path.
        """
        raise NotImplementedError

    # ========================================================
    # SYNCHRONOUS BATCH GENERATION
    # ========================================================

    @abstractmethod
    def generate_batch(
        self,
        texts: list[str],
    ) -> list[SparseEmbedding]:
        """
        Generate sparse representations for multiple texts.

        Used by the document ingestion pipeline.
        """
        raise NotImplementedError