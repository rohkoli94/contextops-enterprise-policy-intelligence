from collections.abc import AsyncGenerator, Generator

from app.db.session import (
    AsyncSessionLocal,
    SessionLocal,
)

from app.dependencies.rag import (
    get_document_chunker,
    get_embedding_provider,
    get_sparse_embedding_provider,
    get_vector_store,
)

from app.providers.llm.factory import (
    get_llm_provider,
)

from app.providers.storage.azure_blob import (
    AzureBlobStorageProvider,
)

from app.rag.ingestion.docling_extractor import (
    DoclingDocumentExtractor,
)

from app.services.document_ingestion_service import (
    DocumentIngestionService,
)

from app.services.document_service import (
    DocumentService,
)


# ================================================================
# SYNCHRONOUS DOCUMENT SERVICE
# ================================================================

def get_document_service() -> Generator[
    DocumentService,
    None,
    None,
]:
    """
    Build the complete synchronous document service
    dependency graph.

    Used by existing synchronous callers.

    Flow:

        SessionLocal
            ->
        Storage
            ->
        LLM Provider
            ->
        Extractor
            ->
        Chunker
            ->
        Dense Embeddings
            ->
        Sparse Embeddings
            ->
        Vector Store
            ->
        Ingestion Service
            ->
        Document Service
    """

    db = SessionLocal()

    try:

        # --------------------------------------------------
        # STORAGE
        # --------------------------------------------------

        storage_provider = (
            AzureBlobStorageProvider()
        )

        # --------------------------------------------------
        # LLM PROVIDER
        # --------------------------------------------------

        llm_provider = (
            get_llm_provider()
        )

        # --------------------------------------------------
        # RAG COMPONENTS
        # --------------------------------------------------

        extractor = (
            DoclingDocumentExtractor(
                llm_provider=llm_provider,
            )
        )

        chunker = (
            get_document_chunker()
        )

        # --------------------------------------------------
        # DENSE EMBEDDING PROVIDER
        # --------------------------------------------------

        embedder = (
            get_embedding_provider()
        )

        # --------------------------------------------------
        # SPARSE BM25 EMBEDDING PROVIDER
        # --------------------------------------------------

        sparse_embedder = (
            get_sparse_embedding_provider()
        )

        # --------------------------------------------------
        # VECTOR STORE
        # --------------------------------------------------

        vector_store = (
            get_vector_store()
        )

        # --------------------------------------------------
        # INGESTION SERVICE
        # --------------------------------------------------

        ingestion_service = (
            DocumentIngestionService(
                storage_provider=storage_provider,
                extractor=extractor,
                chunker=chunker,
                embedder=embedder,
                sparse_embedder=sparse_embedder,
                vector_store=vector_store,
            )
        )

        # --------------------------------------------------
        # DOCUMENT SERVICE
        # --------------------------------------------------

        yield DocumentService(
            db=db,
            storage_provider=storage_provider,
            document_ingestion_service=(
                ingestion_service
            ),
        )

    finally:

        db.close()


# ================================================================
# ASYNCHRONOUS DOCUMENT SERVICE
# ================================================================

async def get_async_document_service() -> AsyncGenerator[
    DocumentService,
    None,
]:
    """
    Build the complete asynchronous document service
    dependency graph.

    Used by FastAPI async document endpoints.

    Flow:

        AsyncSession
            ->
        Async-capable Storage
            ->
        LLM Provider
            ->
        Extractor
            ->
        Chunker
            ->
        Async Dense Embeddings
            ->
        Async Sparse Embeddings
            ->
        Async Vector Store
            ->
        Ingestion Service
            ->
        Document Service

    The individual providers are shared with the existing
    dependency construction functions. Their async methods are
    selected by the asynchronous service path.
    """

    db = AsyncSessionLocal()

    try:

        # --------------------------------------------------
        # STORAGE
        # --------------------------------------------------

        storage_provider = (
            AzureBlobStorageProvider()
        )

        # --------------------------------------------------
        # LLM PROVIDER
        # --------------------------------------------------

        llm_provider = (
            get_llm_provider()
        )

        # --------------------------------------------------
        # RAG COMPONENTS
        # --------------------------------------------------

        extractor = (
            DoclingDocumentExtractor(
                llm_provider=llm_provider,
            )
        )

        chunker = (
            get_document_chunker()
        )

        # --------------------------------------------------
        # DENSE EMBEDDING PROVIDER
        # --------------------------------------------------

        embedder = (
            get_embedding_provider()
        )

        # --------------------------------------------------
        # SPARSE BM25 EMBEDDING PROVIDER
        # --------------------------------------------------

        sparse_embedder = (
            get_sparse_embedding_provider()
        )

        # --------------------------------------------------
        # VECTOR STORE
        # --------------------------------------------------

        vector_store = (
            get_vector_store()
        )

        # --------------------------------------------------
        # INGESTION SERVICE
        # --------------------------------------------------

        ingestion_service = (
            DocumentIngestionService(
                storage_provider=storage_provider,
                extractor=extractor,
                chunker=chunker,
                embedder=embedder,
                sparse_embedder=sparse_embedder,
                vector_store=vector_store,
            )
        )

        # --------------------------------------------------
        # DOCUMENT SERVICE
        # --------------------------------------------------

        yield DocumentService(
            db=db,
            storage_provider=storage_provider,
            document_ingestion_service=(
                ingestion_service
            ),
        )

    finally:

        await db.close()