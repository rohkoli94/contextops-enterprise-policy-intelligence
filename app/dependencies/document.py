from collections.abc import Generator

from app.db.session import SessionLocal
from app.dependencies.rag import (
    get_document_chunker,
    get_embedding_provider,
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
from app.services.document_service import DocumentService


def get_document_ingestion_service(
    storage_provider: AzureBlobStorageProvider,
) -> DocumentIngestionService:
    """
    Build the document ingestion pipeline.

    Dependencies:
        StorageProvider
        DocumentExtractor
        DocumentChunker
        EmbeddingProvider
    """

    return DocumentIngestionService(
        storage_provider=storage_provider,
        extractor=DoclingDocumentExtractor(),
        chunker=get_document_chunker(),
        embedder=get_embedding_provider(),
    )


def get_document_service() -> Generator[
    DocumentService,
    None,
    None,
]:
    """
    Build DocumentService with its RAG ingestion pipeline.
    """

    db = SessionLocal()

    try:
        # Create storage provider once and share it with
        # DocumentService and DocumentIngestionService.
        storage_provider = AzureBlobStorageProvider()

        # Build the complete ingestion pipeline.
        document_ingestion_service = (
            get_document_ingestion_service(
                storage_provider=storage_provider,
            )
        )

        # Inject the ingestion pipeline into DocumentService.
        yield DocumentService(
            db=db,
            storage_provider=storage_provider,
            document_ingestion_service=(
                document_ingestion_service
            ),
        )

    finally:
        db.close()