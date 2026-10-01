import asyncio
from abc import ABC, abstractmethod
from typing import Any

from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)
from app.domain.sparse_embedding import (
    SparseEmbedding,
)
from app.rag.retrieval.models import (
    RetrievedChunk,
)


class VectorStore(ABC):
    """
    Abstraction for vector storage and vector retrieval.

    The application depends on this interface rather than
    directly depending on Qdrant.

    Supports:

        - dense semantic retrieval
        - sparse BM25 retrieval
        - hybrid retrieval
        - synchronous writes
        - asynchronous writes
    """

    # ========================================================
    # COLLECTION
    # ========================================================

    @abstractmethod
    def ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        """
        Create and initialize the vector collection.
        """
        raise NotImplementedError

    # ========================================================
    # SYNCHRONOUS UPSERT
    # ========================================================

    @abstractmethod
    def upsert(
        self,
        chunks: list[EmbeddedDocumentChunk],
    ) -> None:
        """
        Insert or update embedded document chunks.

        Each chunk contains:

            - dense vector
            - sparse BM25 vector
            - document metadata
        """
        raise NotImplementedError

    # ========================================================
    # ASYNCHRONOUS UPSERT
    # ========================================================

    async def aupsert(
        self,
        chunks: list[EmbeddedDocumentChunk],
    ) -> None:
        """
        Insert or update embedded document chunks asynchronously.

        Default implementation delegates the synchronous method
        to a worker thread.

        Concrete vector stores with native async SDK support
        should override this method.
        """

        if chunks is None:
            raise ValueError(
                "Chunks cannot be None."
            )

        await asyncio.to_thread(
            self.upsert,
            chunks,
        )

    # ========================================================
    # DOCUMENT VERSION LIFECYCLE
    # ========================================================

    def set_document_version_status(
        self,
        *,
        document_id: str,
        document_version_id: str,
        tenant_id: str,
        status: str,
    ) -> None:
        """Update the lifecycle status payload for one document version."""
        raise NotImplementedError

    async def aset_document_version_status(
        self,
        *,
        document_id: str,
        document_version_id: str,
        tenant_id: str,
        status: str,
    ) -> None:
        """Async wrapper for document-version status update."""
        await asyncio.to_thread(
            self.set_document_version_status,
            document_id=document_id,
            document_version_id=document_version_id,
            tenant_id=tenant_id,
            status=status,
        )

    def promote_document_version(
        self,
        *,
        document_id: str,
        document_version_id: str,
        previous_active_version_id: str | None,
        tenant_id: str,
    ) -> None:
        """Promote a fully indexed version and retire the previous one."""
        raise NotImplementedError

    async def apromote_document_version(
        self,
        *,
        document_id: str,
        document_version_id: str,
        previous_active_version_id: str | None,
        tenant_id: str,
    ) -> None:
        """Async wrapper for document-version promotion."""
        await asyncio.to_thread(
            self.promote_document_version,
            document_id=document_id,
            document_version_id=document_version_id,
            previous_active_version_id=previous_active_version_id,
            tenant_id=tenant_id,
        )

    # ========================================================
    # DENSE SEARCH
    # ========================================================

    @abstractmethod
    async def asearch_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Asynchronously perform dense semantic retrieval.
        """
        raise NotImplementedError

    # ========================================================
    # SPARSE SEARCH
    # ========================================================

    @abstractmethod
    async def asearch_sparse(
        self,
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Asynchronously perform sparse lexical / BM25 retrieval.
        """
        raise NotImplementedError

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    @abstractmethod
    async def asearch_hybrid(
        self,
        query_vector: list[float],
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Asynchronously perform hybrid dense + sparse retrieval.

        Flow:

            Dense retrieval
                    +
            Sparse/BM25 retrieval
                    Ã¢â€ â€œ
                 Fusion
                    Ã¢â€ â€œ
             Final ranked results
        """
        raise NotImplementedError
