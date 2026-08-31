from app.providers.embedding.base import (
    EmbeddingRequest,
    EmbeddingProvider,
)
from app.providers.vector_store.base import VectorStore
from app.rag.retrieval.base import RetrievalProvider
from app.rag.retrieval.models import RetrievedChunk


class DenseRetriever(RetrievalProvider):
    """
    Dense semantic retriever.

    The query path is asynchronous.

    Flow:

        User Query
            ↓
        EmbeddingProvider
            ↓
        Query Vector
            ↓
        VectorStore.asearch_dense()
            ↓
        Qdrant Dense ANN Search
            ↓
        Top-K Retrieved Chunks
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def aretrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Asynchronously retrieve the most semantically similar
        document chunks.
        """

        # --------------------------------------------------
        # STEP 1 — EMBED QUERY
        # --------------------------------------------------

        embedding_response = (
            await self.embedding_provider.agenerate(
                EmbeddingRequest(
                    text=query,
                )
            )
        )

        query_vector = embedding_response.vector

        # --------------------------------------------------
        # STEP 2 — DENSE SEARCH
        # --------------------------------------------------

        return await self.vector_store.asearch_dense(
            query_vector=query_vector,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )