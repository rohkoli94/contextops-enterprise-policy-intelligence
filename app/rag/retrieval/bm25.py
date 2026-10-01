from app.providers.embedding.sparse_base import (
    SparseEmbeddingProvider,
)
from app.providers.vector_store.base import VectorStore
from app.rag.retrieval.base import RetrievalProvider
from app.rag.retrieval.models import RetrievedChunk


class BM25Retriever(RetrievalProvider):
    """
    Sparse lexical / BM25 retriever.

    The query path is asynchronous.

    Flow:

        User Query
            â†“
        SparseEmbeddingProvider
            â†“
        BM25 Sparse Query
            â†“
        VectorStore.asearch_sparse()
            â†“
        Qdrant BM25 Search
            â†“
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

    async def aretrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Asynchronously retrieve the most lexically relevant
        document chunks using BM25 sparse retrieval.
        """

        # --------------------------------------------------
        # STEP 1 â€” GENERATE BM25 SPARSE QUERY
        # --------------------------------------------------

        sparse_query = (
            await self.sparse_embedding_provider.agenerate(
                query
            )
        )

        # --------------------------------------------------
        # STEP 2 â€” BM25 SEARCH
        # --------------------------------------------------

        return await self.vector_store.asearch_sparse(
            sparse_query=sparse_query,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )
