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
             VectorStore.search_hybrid()
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

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant document chunks using both:

        - dense semantic retrieval
        - sparse BM25 lexical retrieval

        The final fusion is delegated to the VectorStore
        implementation.
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
        # STEP 1 — GENERATE DENSE QUERY
        # --------------------------------------------------

        embedding_response = (
            self.embedding_provider.generate(
                EmbeddingRequest(
                    text=query,
                )
            )
        )

        query_vector = (
            embedding_response.vector
        )

        # --------------------------------------------------
        # STEP 2 — GENERATE BM25 SPARSE QUERY
        # --------------------------------------------------

        sparse_query = (
            self.sparse_embedding_provider.generate(
                query
            )
        )

        # --------------------------------------------------
        # STEP 3 — HYBRID SEARCH
        # --------------------------------------------------

        return self.vector_store.search_hybrid(
            query_vector=query_vector,
            sparse_query=sparse_query,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )