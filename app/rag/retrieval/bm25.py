from app.providers.embedding.sparse_base import (
    SparseEmbeddingProvider,
)
from app.providers.vector_store.base import VectorStore
from app.rag.retrieval.base import RetrievalProvider
from app.rag.retrieval.models import RetrievedChunk


class BM25Retriever(RetrievalProvider):
    """
    Sparse lexical / BM25 retriever.

    Flow:

        User Query
            ↓
        SparseEmbeddingProvider
            ↓
        BM25 Sparse Query
            ↓
        VectorStore.search_sparse()
            ↓
        Qdrant BM25 Search
            ↓
        Top-K Retrieved Chunks
    """

    def __init__(
        self,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.sparse_embedding_provider = (
            sparse_embedding_provider
        )
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most lexically relevant document chunks
        using BM25 sparse retrieval.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # --------------------------------------------------
        # STEP 1 — GENERATE BM25 SPARSE QUERY
        # --------------------------------------------------

        sparse_query = (
            self.sparse_embedding_provider.generate(
                query
            )
        )

        # --------------------------------------------------
        # STEP 2 — BM25 SEARCH
        # --------------------------------------------------

        return self.vector_store.search_sparse(
            sparse_query=sparse_query,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )