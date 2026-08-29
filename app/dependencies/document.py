from collections.abc import Generator

from app.db.session import SessionLocal

from app.dependencies.rag import (
    get_document_chunker,
    get_embedding_provider,
    get_sparse_embedding_provider,
    get_vector_store,
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


def get_document_service() -> Generator[
    DocumentService,
    None,
    None,
]:
    """
    Build the complete document service dependency graph.
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
        # RAG COMPONENTS
        # --------------------------------------------------

        extractor = (
            DoclingDocumentExtractor()
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