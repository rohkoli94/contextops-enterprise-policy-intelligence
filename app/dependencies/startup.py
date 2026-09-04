from fastapi import FastAPI

from app.dependencies.container import (
    create_query_service,
)

from app.dependencies.rag import (
    get_embedding_provider,
    get_sparse_embedding_provider,
    get_vector_store,
    get_hybrid_retriever,
)


# ============================================================
# RAG INITIALIZATION
# ============================================================

def initialize_rag() -> None:
    """
    Initialize shared RAG infrastructure during application
    startup.

    Initialization order:

        1. Dense embedding provider
        2. Embedding vector dimension
        3. Sparse BM25 provider
        4. Hybrid retriever
        5. Vector store
        6. Qdrant collection
    """

    # --------------------------------------------------------
    # STEP 1 — DENSE EMBEDDING PROVIDER
    # --------------------------------------------------------

    embedding_provider = (
        get_embedding_provider()
    )

    # --------------------------------------------------------
    # STEP 2 — VECTOR DIMENSION
    # --------------------------------------------------------

    vector_size = (
        embedding_provider.get_dimension()
    )

    # --------------------------------------------------------
    # STEP 3 — SPARSE EMBEDDING PROVIDER
    # --------------------------------------------------------

    get_sparse_embedding_provider()

    # --------------------------------------------------------
    # STEP 4 — HYBRID RETRIEVER
    # --------------------------------------------------------

    get_hybrid_retriever()

    # --------------------------------------------------------
    # STEP 5 — VECTOR STORE
    # --------------------------------------------------------

    vector_store = get_vector_store()

    # --------------------------------------------------------
    # STEP 6 — QDRANT COLLECTION
    # --------------------------------------------------------

    vector_store.ensure_collection(
        vector_size=vector_size,
    )


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

def initialize_application(
    app: FastAPI,
) -> None:
    """
    Initialize all shared application infrastructure once
    during FastAPI startup.

    Long-lived components are created here and stored in
    application state.

    Request-specific objects are created later per request.
    """

    # --------------------------------------------------------
    # STEP 1 — RAG INFRASTRUCTURE
    # --------------------------------------------------------

    initialize_rag()

    # --------------------------------------------------------
    # STEP 2 — QUERY SERVICE
    # --------------------------------------------------------
    #
    # create_query_service() performs the complete composition:
    #
    #   LLM Provider
    #   Hybrid Retriever
    #   Conversation Memory
    #   Query Rewriter
    #   Cache
    #   Guardrails
    #   Reranker
    #   Retrieval Validator
    #   Grounding Validator
    #           ↓
    #      LangGraph
    #           ↓
    #      QueryService
    #
    # Everything is created once during application startup.

    query_service = (
        create_query_service()
    )

    # --------------------------------------------------------
    # STORE SHARED QUERY SERVICE
    # --------------------------------------------------------

    app.state.query_service = (
        query_service
    )

    # --------------------------------------------------------
    # STORE SHARED QUERY GRAPH
    # --------------------------------------------------------
    #
    # The QueryService now owns the compiled LangGraph.
    #
    # Keeping the graph separately available in application
    # state is useful for:
    #
    #   - observability
    #   - debugging
    #   - future graph inspection
    #   - dependency injection
    #
    # The QueryService remains the normal application entry point.

    app.state.query_graph = (
        query_service.query_graph
    )