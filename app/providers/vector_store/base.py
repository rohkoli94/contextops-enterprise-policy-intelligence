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

    The vector store supports:

        - dense semantic retrieval
        - sparse BM25 lexical retrieval
        - hybrid retrieval
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
    # UPSERT
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
    # DENSE SEARCH
    # ========================================================

    @abstractmethod
    def search_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform dense semantic retrieval.

        query_vector:
            Dense embedding of the user query.

        tenant_id:
            Tenant whose data may be searched.

        top_k:
            Maximum number of results.

        filters:
            Optional metadata filters such as:

                - document_id
                - document_version_id
                - categories
                - tags
                - content_type
        """
        raise NotImplementedError

    # ========================================================
    # SPARSE SEARCH
    # ========================================================

    @abstractmethod
    def search_sparse(
        self,
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform sparse lexical / BM25 retrieval.

        sparse_query:
            BM25 sparse representation of the user query.

        tenant_id:
            Tenant whose data may be searched.

        top_k:
            Maximum number of results.

        filters:
            Optional metadata filters.
        """
        raise NotImplementedError

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    @abstractmethod
    def search_hybrid(
        self,
        query_vector: list[float],
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform hybrid dense + sparse retrieval.

        query_vector:
            Dense embedding of the user query.

        sparse_query:
            BM25 sparse representation of the user query.

        tenant_id:
            Tenant whose data may be searched.

        top_k:
            Maximum number of final results.

        filters:
            Optional metadata filters.

        The implementation is responsible for:

            dense candidate retrieval
                    +
            sparse/BM25 candidate retrieval
                    ↓
                   fusion
                    ↓
              final ranked results
        """
        raise NotImplementedError