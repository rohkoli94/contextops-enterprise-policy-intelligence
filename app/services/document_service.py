import uuid
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.api.v1.documents.schemas.document_upload_response import (
    DocumentUploadResponse,
)
from app.models.category import Category
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.tag import Tag
from app.providers.storage.base import StorageProvider
from app.utils.hashing import calculate_stream_hash


class DocumentService:
    def __init__(
        self,
        db: Session,
        storage_provider: StorageProvider,
    ):
        self.db = db
        self.storage_provider = storage_provider

    def upload_document(
        self,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        document_name: str,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> DocumentUploadResponse:
        document_id = uuid.uuid4()
        document_version_id = uuid.uuid4()
        version = 1
        stored_blob_path = None

        # Calculate SHA-256 and file size in chunks.
        # The complete file is not loaded into memory.
        content_hash, file_size = calculate_stream_hash(stream)

        document = Document(
            document_id=document_id,
            document_name=document_name,
            current_version=version,
            status="PROCESSING",
        )

        self.db.add(document)

        blob_path = (
            f"documents/{document_id}/"
            f"v{version}/{file_name}"
        )

        try:
            # Upload file to Azure Blob Storage.
            stored_blob_path = self.storage_provider.upload(
                path=blob_path,
                stream=stream,
                content_type=content_type,
            )

            # Create version-specific database record.
            document_version = DocumentVersion(
                document_version_id=document_version_id,
                document_id=document_id,
                version=version,
                file_name=file_name,
                content_type=content_type,
                file_size=file_size,
                content_hash=content_hash,
                blob_path=stored_blob_path,
            )

            self.db.add(document_version)

            # Create or reuse categories.
            if categories:
                for category_name in categories:
                    category = (
                        self.db.query(Category)
                        .filter(Category.name == category_name)
                        .first()
                    )

                    if category is None:
                        category = Category(name=category_name)
                        self.db.add(category)

                    document.categories.append(category)

            # Create or reuse tags.
            if tags:
                for tag_name in tags:
                    tag = (
                        self.db.query(Tag)
                        .filter(Tag.name == tag_name)
                        .first()
                    )

                    if tag is None:
                        tag = Tag(name=tag_name)
                        self.db.add(tag)

                    document.tags.append(tag)

            # Commit all PostgreSQL changes.
            self.db.commit()

            return DocumentUploadResponse(
                document_id=document_id,
                document_version_id=document_version_id,
                document_name=document_name,
                version=version,
                status="PROCESSING"
            )

        except Exception:
            # Roll back PostgreSQL changes.
            self.db.rollback()

            # Compensating cleanup:
            # If Blob upload succeeded but DB processing failed,
            # attempt to remove the uploaded Blob.
            if stored_blob_path:
                try:
                    self.storage_provider.delete(stored_blob_path)
                except Exception:
                    # Preserve the original exception.
                    pass

            raise