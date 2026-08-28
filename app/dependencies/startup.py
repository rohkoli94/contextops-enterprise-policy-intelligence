from app.dependencies.rag import (
    get_embedding_provider,
    get_vector_store,
)


def initialize_qdrant() -> None:
    """
    Initialize Qdrant infrastructure when the application starts.

    Steps:

        1. Create embedding provider
        2. Get embedding vector dimension
        3. Create vector store
        4. Initialize Qdrant collection
    """

    # --------------------------------------------------
    # STEP 1 — EMBEDDING PROVIDER
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
    # STEP 3 — VECTOR STORE
    # --------------------------------------------------

    vector_store = get_vector_store()

    # --------------------------------------------------
    # STEP 4 — INITIALIZE QDRANT
    # --------------------------------------------------

    vector_store.ensure_collection(
        vector_size=vector_size,
    )