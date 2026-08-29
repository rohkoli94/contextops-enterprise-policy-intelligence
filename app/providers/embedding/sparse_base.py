from abc import ABC, abstractmethod

from app.domain.sparse_embedding import SparseEmbedding


class SparseEmbeddingProvider(ABC):
    """
    Abstraction for sparse embedding generation.

    Implementations can generate lexical/sparse
    representations such as BM25.
    """

    @abstractmethod
    def generate(
        self,
        text: str,
    ) -> SparseEmbedding:
        """
        Generate a sparse representation for one text.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self,
        texts: list[str],
    ) -> list[SparseEmbedding]:
        """
        Generate sparse representations for multiple texts.
        """
        raise NotImplementedError