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

    Flow:

        User Query
            ↓
        EmbeddingProvider
            ↓
        Query Vector
            ↓
        VectorStore
            ↓
        Qdrant ANN Search
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

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most semantically similar document chunks.
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
        # STEP 1 — EMBED QUERY
        # --------------------------------------------------

        embedding_response = (
            self.embedding_provider.generate(
                EmbeddingRequest(
                    text=query,
                )
            )
        )

        query_vector = embedding_response.vector

        # --------------------------------------------------
        # STEP 2 — SEARCH VECTOR STORE
        # --------------------------------------------------

        return self.vector_store.search(
            query_vector=query_vector,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )