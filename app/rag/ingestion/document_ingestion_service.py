from app.domain.document import Document
from app.domain.document_element import DocumentElement
from app.providers.storage.base import StorageProvider
from app.rag.ingestion.base import DocumentExtractor


class DocumentIngestionService:
    """
    Orchestrates document ingestion.

    Current flow:

        Stored document
            ->
        DocumentExtractor
            ->
        DocumentElements

    Future flow:

        Extract
            ->
        Chunk
            ->
        Embed
            ->
        Vector DB
    """

    def __init__(
        self,
        storage_provider: StorageProvider,
        extractor: DocumentExtractor,
    ) -> None:
        self.storage_provider = storage_provider
        self.extractor = extractor

    def ingest(
        self,
        document: Document,
        blob_path: str,
        file_name: str,
    ) -> list[DocumentElement]:
        """
        Read the persisted document from storage and extract
        structured document elements.
        """

        stream = self.storage_provider.download(
            path=blob_path,
        )

        try:
            return self.extractor.extract(
                document=document,
                stream=stream,
                file_name=file_name,
            )

        finally:
            stream.close()