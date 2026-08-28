from abc import ABC, abstractmethod
from typing import Any

from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)

from app.rag.retrieval.models import RetrievedChunk


class VectorStore(ABC):
    """
    Abstraction for vector storage and vector retrieval.

    The application depends on this interface rather than
    directly depending on Qdrant.
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
        Insert or update embedded document chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[EmbeddedDocumentChunk]:
        """
        Search for vectors relevant to a query.

        tenant_id:
            Tenant whose data may be searched.

        top_k:
            Maximum number of results.

        filters:
            Optional metadata filters such as categories,
            tags, document version, content type, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
            Search for vectors relevant to a query.
    
            tenant_id:
                Tenant whose data may be searched.
    
            top_k:
                Maximum number of results.
    
            filters:
                Optional metadata filters such as categories,
                tags, document version, content type, etc.
        """
        raise NotImplementedError