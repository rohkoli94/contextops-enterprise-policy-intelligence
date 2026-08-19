import shutil
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO

from docling.document_converter import DocumentConverter

from app.domain.document import Document
from app.domain.document_element import (
    ContentType,
    DocumentElement,
)
from app.rag.ingestion.base import DocumentExtractor
from app.utils.hashing import calculate_content_hash


class DoclingDocumentExtractor(DocumentExtractor):
    """
    Extracts structured content from PDF documents using Docling.
    """

    def __init__(self) -> None:
        # Create the Docling converter once and reuse it.
        self.converter = DocumentConverter()

    def extract(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str
    ) -> list[DocumentElement]:
        # Keep track of the temporary file so it can be deleted
        # after document processing.
        temp_file_path: Path | None = None

        try:
            suffix = Path(file_name).suffix or ".tmp"

            # Docling processes the PDF from a file path.
            # Copy the incoming stream to a temporary file.
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:
                # Copy the file incrementally using a 1 MB buffer.
                # This avoids loading the entire PDF into RAM at once.
                shutil.copyfileobj(
                    stream,
                    temp_file,
                    length=1024 * 1024,
                )

                temp_file_path = Path(temp_file.name)

            # Parse the PDF using Docling.
            result = self.converter.convert(temp_file_path)

            elements: list[DocumentElement] = []

            # Iterate through the structured items extracted by Docling.
            for item, _level in result.document.iterate_items():
                # For the current implementation, process items
                # that contain textual content.
                if not hasattr(item, "text"):
                    continue

                content = item.text.strip()

                # Skip empty content.
                if not content:
                    continue

                # Default page number.
                page_number = 1

                # Docling provenance contains the source page information.
                if item.prov:
                    page_number = item.prov[0].page_no

                # Convert the extracted content into our domain model.
                elements.append(
                    DocumentElement(
                        element_id=str(uuid.uuid4()),
                        document_id=document.document_id,
                        page_number=page_number,
                        content_type=ContentType.TEXT,
                        content=content,
                        content_hash=calculate_content_hash(
                            content
                        ),
                    )
                )

            # Return all extracted document elements.
            return elements

        finally:
            # Always delete the temporary file, including when
            # document parsing fails.
            if temp_file_path is not None:
                temp_file_path.unlink(missing_ok=True)


# ============================================================
# rohit notes:—
# ============================================================

# 1. Technical challenge:
# Large PDF files, potentially up to 1 GB, should not be loaded
# entirely into application memory during ingestion.

# 2. Current solution:
# The Azure/storage stream is copied to a temporary file using a
# 1 MB buffer, so only a small chunk is held in RAM at a time.

# 3. Area of improvement:
# The complete file is temporarily stored on disk.
# Using BytesIO could avoid temporary disk storage, but a 1 GB file
# could then require approximately 1 GB of RAM, which is not suitable
# for large-file ingestion. Future improvements could use a scalable
# file-processing or worker-based architecture for very large documents.

# 4. Future improvement:
# Move document extraction to asynchronous background processing.
# The API can return immediately after creating an ingestion job,
# while a separate worker processes:
# Download → Docling Extraction → Chunking → Embedding → Vector DB.
#
# This prevents long-running document processing from blocking
# the FastAPI request and improves scalability for large documents.