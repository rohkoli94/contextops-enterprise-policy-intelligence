import uuid
from typing import BinaryIO

from sqlalchemy.orm import Session, selectinload

from app.api.v1.documents.schemas.document_list_response import (
    DocumentListItem,
    DocumentListResponse,
)
from app.api.v1.documents.schemas.document_upload_response import (
    DocumentUploadResponse,
)
from app.models.category import Category
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.tag import Tag
from app.providers.storage.base import StorageProvider
from app.services.document_ingestion_service import (
    DocumentIngestionService,
)
from app.utils.hashing import calculate_stream_hash


class DocumentService:
    """
    Handles document lifecycle operations.

    Responsibilities:
    - Create documents
    - Upload document versions
    - Store original files
    - Manage categories and tags
    - Trigger the RAG ingestion pipeline
    """

    def __init__(
        self,
        db: Session,
        storage_provider: StorageProvider,
        document_ingestion_service: DocumentIngestionService,
    ) -> None:
        self.db = db
        self.storage_provider = storage_provider
        self.document_ingestion_service = (
            document_ingestion_service
        )

    def upload_document(
        self,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        document_name: str,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> DocumentUploadResponse:
        """
        Upload a new document.

        Flow:

        Create Document
            ->
        Add categories and tags
            ->
        Calculate hash and file size
            ->
        Upload Version 1
            ->
        Trigger ingestion
        """

        document_id = uuid.uuid4()

        document = Document(
            document_id=document_id,
            document_name=document_name,
            current_version=0,
            status="PROCESSING",
        )

        self.db.add(document)

        self._add_categories(
            document=document,
            categories=categories,
        )

        self._add_tags(
            document=document,
            tags=tags,
        )

        # Calculate the content hash and file size.
        #
        # calculate_stream_hash should restore the stream position
        # after reading it so the same stream can be uploaded.
        content_hash, file_size = calculate_stream_hash(stream)

        return self._upload_document_version(
            document=document,
            stream=stream,
            file_name=file_name,
            content_type=content_type,
            version=1,
            content_hash=content_hash,
            file_size=file_size,
        )

    def upload_new_version(
        self,
        document_id: uuid.UUID,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
    ) -> DocumentUploadResponse:
        """
        Upload a new version of an existing document.

        Flow:

        Find Document
            ->
        Calculate hash
            ->
        Check duplicate version
            ->
        Create next version
            ->
        Upload
            ->
        Trigger ingestion
        """

        document = (
            self.db.query(Document)
            .filter(Document.document_id == document_id)
            .first()
        )

        if document is None:
            raise ValueError("Document not found")

        # Calculate the content hash and file size.
        content_hash, file_size = calculate_stream_hash(stream)

        # Prevent uploading identical document content again.
        existing_version = (
            self.db.query(DocumentVersion)
            .filter(
                DocumentVersion.document_id == document_id,
                DocumentVersion.content_hash == content_hash,
            )
            .first()
        )

        if existing_version is not None:
            raise ValueError(
                "An identical document version already exists"
            )

        # Determine the next version number.
        version = document.current_version + 1

        return self._upload_document_version(
            document=document,
            stream=stream,
            file_name=file_name,
            content_type=content_type,
            version=version,
            content_hash=content_hash,
            file_size=file_size,
        )

    def _upload_document_version(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        version: int,
        content_hash: str,
        file_size: int,
    ) -> DocumentUploadResponse:
        """
        Shared upload flow used by both:

        - upload_document()
        - upload_new_version()

        Flow:

        Upload original file to storage
            ->
        Create DocumentVersion
            ->
        Update current version
            ->
        Commit database transaction
            ->
        Trigger RAG ingestion
        """

        document_version_id = uuid.uuid4()
        stored_blob_path: str | None = None

        # Keep every document version separately in storage.
        #
        # Example:
        # documents/<document-id>/v1/policy.pdf
        blob_path = (
            f"documents/{document.document_id}/"
            f"v{version}/{file_name}"
        )

        try:
            # Store the original document.
            stored_blob_path = self.storage_provider.upload(
                path=blob_path,
                stream=stream,
                content_type=content_type,
            )

            # Create the database record for this version.
            document_version = DocumentVersion(
                document_version_id=document_version_id,
                document_id=document.document_id,
                version=version,
                file_name=file_name,
                content_type=content_type,
                file_size=file_size,
                content_hash=content_hash,
                blob_path=stored_blob_path,
            )

            self.db.add(document_version)

            # Update the latest version number.
            document.current_version = version

            # Persist the document and version before ingestion.
            #
            # This ensures the original document and its version
            # are successfully stored before RAG processing begins.
            self.db.commit()

            # Read the persisted document from storage and start
            # the RAG ingestion pipeline.
            #
            # Current ingestion:
            #
            # Storage
            #   ->
            # Docling Extraction
            #   ->
            # TEXT / TABLE / IMAGE / CHART / DIAGRAM
            #
            # Future:
            #
            # Extraction
            #   ->
            # Chunking
            #   ->
            # Embedding
            #   ->
            # Vector DB
            self.document_ingestion_service.ingest(
                document=document,
                blob_path=stored_blob_path,
                file_name=file_name,
            )

            return DocumentUploadResponse(
                document_id=document.document_id,
                document_version_id=document_version_id,
                document_name=document.document_name,
                version=version,
                status=document.status,
            )

        except Exception:
            # Roll back uncommitted database changes.
            self.db.rollback()

            # If storage upload succeeded but a failure happened
            # before successful processing, attempt cleanup.
            if stored_blob_path:
                try:
                    self.storage_provider.delete(
                        stored_blob_path
                    )
                except Exception:
                    # Do not hide the original exception if cleanup fails.
                    pass

            raise

    def _add_categories(
        self,
        document: Document,
        categories: list[str] | None,
    ) -> None:
        """
        Attach categories to the document.

        Reuses an existing category when available;
        otherwise creates a new one.
        """

        if not categories:
            return

        for category_name in categories:
            category = (
                self.db.query(Category)
                .filter(Category.name == category_name)
                .first()
            )

            if category is None:
                category = Category(
                    name=category_name,
                )
                self.db.add(category)

            document.categories.append(category)

    def _add_tags(
        self,
        document: Document,
        tags: list[str] | None,
    ) -> None:
        """
        Attach tags to the document.

        Reuses an existing tag when available;
        otherwise creates a new one.
        """

        if not tags:
            return

        for tag_name in tags:
            tag = (
                self.db.query(Tag)
                .filter(Tag.name == tag_name)
                .first()
            )

            if tag is None:
                tag = Tag(
                    name=tag_name,
                )
                self.db.add(tag)

            document.tags.append(tag)

    def get_active_documents(
        self,
    ) -> DocumentListResponse:
        """
        Return all active, non-deleted documents.
        """

        documents = (
            self.db.query(Document)
            .options(
                selectinload(Document.categories),
                selectinload(Document.tags),
            )
            .filter(
                Document.deleted_at.is_(None)
            )
            .order_by(Document.created_at.desc())
            .all()
        )

        return DocumentListResponse(
            documents=[
                DocumentListItem(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    current_version=document.current_version,
                    status=document.status,
                    categories=[
                        category.name
                        for category in document.categories
                    ],
                    tags=[
                        tag.name
                        for tag in document.tags
                    ],
                    created_at=document.created_at,
                )
                for document in documents
            ]
        )


# ============================================================
# ROHIT NOTES —
# ============================================================

# Why is ingestion triggered from _upload_document_version()?
#
# Both upload_document() and upload_new_version() eventually use
# the same shared method.
#
# upload_document()
#       |
#       └──────> _upload_document_version()
#                       |
# upload_new_version()  |
#       |                |
#       └────────────────┘
#                       ↓
#              Store document version
#                       ↓
#              Trigger ingestion
#
# This avoids duplicating ingestion logic in both methods.


# Why commit before ingestion?
#
# We first ensure that:
#
# 1. The original file is stored successfully.
# 2. The DocumentVersion record is persisted.
# 3. document.current_version is updated.
#
# Only after that do we start RAG ingestion.
#
# This ensures ingestion works against a persisted document.


# Why does ingestion read from Blob Storage?
#
# The original upload stream may already have been read for:
#
# - Hash calculation
# - File size calculation
# - Storage upload
#
# Instead of depending on stream position, Blob Storage becomes
# the single source of truth:
#
# Upload
#   ->
# Persist to Blob Storage
#   ->
# Download persisted file
#   ->
# RAG ingestion


# Current flow:
#
# POST /documents
#       |
#       ↓
# upload_document()
#       |
#       ↓
# _upload_document_version()
#       |
#       ↓
# Blob Storage
#       |
#       ↓
# DocumentIngestionService
#       |
#       ↓
# DoclingDocumentExtractor
#       |
#       ├── TEXT
#       ├── TABLE
#       └── Visual
#              |
#              ↓
#        Vision-capable LLM
#              |
#              ↓
#       IMAGE / CHART / DIAGRAM
#
#
# POST /documents/{document_id}/versions follows the same
# _upload_document_version() -> ingestion pipeline.


# Future improvement:
#
# Currently ingestion runs synchronously after upload.
#
# For large PDFs and vision processing, this should eventually move
# to a background worker architecture:
#
# API
#   ->
# Store Document + Version
#   ->
# Create Ingestion Job
#   ->
# Return response immediately
#
# Worker
#   ->
# Download from Blob
#   ->
# Extract
#   ->
# Vision Analysis
#   ->
# Chunk
#   ->
# Embed
#   ->
# Vector DB
#
# This prevents long-running RAG processing from blocking the API.