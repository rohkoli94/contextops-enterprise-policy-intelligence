from abc import ABC, abstractmethod

from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)


class VectorStore(ABC):
    """
    Abstraction for vector storage.

    Application services depend on this interface,
    not on a specific vector database implementation.
    """

    @abstractmethod
    def ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        """
        Create and initialize the vector collection.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        chunks: list[EmbeddedDocumentChunk],
    ) -> None:
        """
        Insert or update embedded chunks.
        """
        raise NotImplementedError