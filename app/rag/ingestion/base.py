import asyncio
from abc import ABC, abstractmethod
from typing import BinaryIO

from app.domain.document import Document
from app.domain.document_element import DocumentElement


class DocumentExtractor(ABC):
    """
    Abstraction for document extraction.

    The synchronous extract() method remains the core contract
    because libraries such as Docling perform CPU/blocking work.

    aextract() provides the preferred interface for the async
    ingestion pipeline. The default implementation executes the
    synchronous extractor in a worker thread so the FastAPI event
    loop is not blocked.

    Extractor implementations with a genuinely asynchronous
    extraction engine can override aextract().
    """

    # ============================================================
    # SYNCHRONOUS EXTRACTION
    # ============================================================

    @abstractmethod
    def extract(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str,
        document_version_id,
    ) -> list[DocumentElement]:
        """
        Extract structured document elements synchronously.

        Implementations may return elements representing:

        - text
        - tables
        - images
        - charts
        - diagrams
        """
        raise NotImplementedError

    # ============================================================
    # ASYNCHRONOUS EXTRACTION
    # ============================================================

    async def aextract(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str,
        document_version_id,
    ) -> list[DocumentElement]:
        """
        Extract structured document elements asynchronously.

        The default implementation runs the synchronous extractor
        in a worker thread.

        This is important for FastAPI because document extraction
        can be CPU-intensive and blocking.

        Native asynchronous extractor implementations may override
        this method.
        """

        return await asyncio.to_thread(
            self.extract,
            document,
            stream,
            file_name,
            document_version_id,
        )