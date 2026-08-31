from fastapi import FastAPI, Request

from app.dependencies.container import (
    create_query_service,
)

from app.dependencies.rag import (
    get_embedding_provider,
    get_sparse_embedding_provider,
    get_vector_store,
    get_hybrid_retriever,
)


def initialize_rag() -> None:
    """
    Initialize RAG infrastructure when the application starts.

    Steps:

        1. Create dense embedding provider
        2. Get embedding vector dimension
        3. Initialize sparse BM25 provider
        4. Initialize hybrid retriever
        5. Create vector store
        6. Initialize Qdrant collection
    """

    # --------------------------------------------------
    # STEP 1 — DENSE EMBEDDING PROVIDER
    # --------------------------------------------------

    embedding_provider = (
        get_embedding_provider()
    )

    # --------------------------------------------------
    # STEP 2 — GET VECTOR DIMENSION
    # --------------------------------------------------

    vector_size = (
        embedding_provider.get_dimension()
    )

    # --------------------------------------------------
    # STEP 3 — SPARSE BM25 PROVIDER
    # --------------------------------------------------

    # Force initialization of the shared BM25 provider
    # during application startup.
    get_sparse_embedding_provider()

    # --------------------------------------------------
    # STEP 4 — HYBRID RETRIEVER
    # --------------------------------------------------

    # Force construction of the shared hybrid retriever
    # during application startup.
    get_hybrid_retriever()

    # --------------------------------------------------
    # STEP 5 — VECTOR STORE
    # --------------------------------------------------

    vector_store = get_vector_store()

    # --------------------------------------------------
    # STEP 6 — INITIALIZE QDRANT
    # --------------------------------------------------

    vector_store.ensure_collection(
        vector_size=vector_size,
    )


def initialize_application(
    app: FastAPI,
) -> None:
    """
    Initialize shared application infrastructure and services.

    All long-lived components are created once during
    application startup and stored in application state.
    """

    # --------------------------------------------------
    # STEP 1 — RAG INFRASTRUCTURE
    # --------------------------------------------------

    initialize_rag()

    # --------------------------------------------------
    # STEP 2 — QUERY SERVICE
    # --------------------------------------------------

    # QueryService is composed once during application startup.
    #
    # It reuses the shared:
    #
    # - Microsoft Foundry LLM provider
    # - HybridRetriever
    #
    # The same QueryService instance is reused by
    # subsequent HTTP requests.

    app.state.query_service = (
        create_query_service()
    )


def get_query_service(
    request: Request,
):
    """
    Return the application-scoped QueryService.

    The QueryService is created once during startup and
    retrieved from FastAPI application state for each request.
    """

    return request.app.state.query_service