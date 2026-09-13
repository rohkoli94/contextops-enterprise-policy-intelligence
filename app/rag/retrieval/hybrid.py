import asyncio

from app.providers.embedding.base import (
    EmbeddingRequest,
    EmbeddingProvider,
)
from app.providers.embedding.sparse_base import (
    SparseEmbeddingProvider,
)
from app.providers.vector_store.base import VectorStore
from app.rag.retrieval.base import RetrievalProvider
from app.rag.retrieval.models import RetrievedChunk


class HybridRetriever(RetrievalProvider):
    """
    Hybrid retriever combining dense semantic retrieval
    with sparse BM25 lexical retrieval.

    The production query path is asynchronous.

    Flow:

        User Query
            ↓
        ┌──────────────────────────┐
        │                          │
        ↓                          ↓
    Dense Provider          Sparse BM25 Provider
        ↓                          ↓
    Dense Query              Sparse Query
        └────────────┬─────────────┘
                     ↓
             VectorStore.asearch_hybrid()
                     ↓
                  Qdrant
                     ↓
             Dense + BM25 Search
                     ↓
                    RRF
                     ↓
             RetrievedChunk[]
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider

        self.sparse_embedding_provider = (
            sparse_embedding_provider
        )

        self.vector_store = vector_store

    # ========================================================
    # SYNCHRONOUS HYBRID RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Synchronously retrieve relevant document chunks.

        The actual implementation lives in aretrieve() so that
        both interfaces use exactly the same retrieval logic.

        This method satisfies the synchronous RetrievalProvider
        contract.

        The synchronous method must not be called while another
        asyncio event loop is already running. Async callers
        should use aretrieve() directly.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve(
                    query=query,
                    tenant_id=tenant_id,
                    top_k=top_k,
                    filters=filters,
                )
            )

        raise RuntimeError(
            "HybridRetriever.retrieve() cannot be called "
            "from a running event loop. Use aretrieve() "
            "instead."
        )

    # ========================================================
    # ASYNC HYBRID RETRIEVAL
    # ========================================================

    async def aretrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Asynchronously retrieve relevant document chunks
        using both dense semantic retrieval and sparse
        BM25 lexical retrieval.

        The final fusion is delegated to the VectorStore.
        """

        # ----------------------------------------------------
        # STEP 1 — GENERATE DENSE QUERY
        # ----------------------------------------------------

        embedding_response = (
            await self.embedding_provider.agenerate(
                EmbeddingRequest(
                    text=query,
                )
            )
        )

        query_vector = (
            embedding_response.vector
        )

        # ----------------------------------------------------
        # STEP 2 — GENERATE BM25 SPARSE QUERY
        # ----------------------------------------------------

        sparse_query = (
            await self.sparse_embedding_provider.agenerate(
                query
            )
        )

        # ----------------------------------------------------
        # STEP 3 — HYBRID SEARCH
        # ----------------------------------------------------

        return await self.vector_store.asearch_hybrid(
            query_vector=query_vector,
            sparse_query=sparse_query,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )