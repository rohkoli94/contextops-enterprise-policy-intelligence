import asyncio
import uuid
from typing import BinaryIO

from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
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

    def __init__(
        self,
        db: Session | AsyncSession,
        storage_provider: StorageProvider,
        document_ingestion_service: DocumentIngestionService,
    ) -> None:
        self.db = db
        self.storage_provider = storage_provider
        self.document_ingestion_service = (
            document_ingestion_service
        )

    # ============================================================
    # SYNCHRONOUS
    # ============================================================

    def upload_document(
        self,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        document_name: str,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> DocumentUploadResponse:

        if isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "upload_document() requires a synchronous "
                "SQLAlchemy Session. Use aupload_document()."
            )

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

        content_hash, file_size = (
            calculate_stream_hash(stream)
        )

        return self._upload_document_version(
            document=document,
            stream=stream,
            file_name=file_name,
            content_type=content_type,
            version=1,
            content_hash=content_hash,
            file_size=file_size,
        )

    # ============================================================
    # ASYNCHRONOUS
    # ============================================================

    async def aupload_document(
        self,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        document_name: str,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        background_tasks: BackgroundTasks | None = None,
    ) -> DocumentUploadResponse:

        if not isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "aupload_document() requires an AsyncSession."
            )

        document_id = uuid.uuid4()

        document = Document(
            document_id=document_id,
            document_name=document_name,
            current_version=0,
            status="PROCESSING",
        )

        self.db.add(document)

        await self._aadd_categories(
            document=document,
            categories=categories,
        )

        await self._aadd_tags(
            document=document,
            tags=tags,
        )

        content_hash, file_size = (
            await asyncio.to_thread(
                calculate_stream_hash,
                stream,
            )
        )

        return await self._aupload_document_version(
            document=document,
            stream=stream,
            file_name=file_name,
            content_type=content_type,
            version=1,
            content_hash=content_hash,
            file_size=file_size,
            background_tasks=background_tasks,
        )

    # ============================================================
    # SYNCHRONOUS NEW VERSION
    # ============================================================

    def upload_new_version(
        self,
        document_id: uuid.UUID,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
    ) -> DocumentUploadResponse:

        if isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "upload_new_version() requires a synchronous "
                "SQLAlchemy Session. Use aupload_new_version()."
            )

        document = (
            self.db.query(Document)
            .filter(
                Document.document_id == document_id
            )
            .first()
        )

        if document is None:
            raise ValueError("Document not found")

        content_hash, file_size = (
            calculate_stream_hash(stream)
        )

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

    # ============================================================
    # ASYNCHRONOUS NEW VERSION
    # ============================================================

    async def aupload_new_version(
        self,
        document_id: uuid.UUID,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        background_tasks: BackgroundTasks | None = None,
    ) -> DocumentUploadResponse:

        if not isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "aupload_new_version() requires an AsyncSession."
            )

        result = await self.db.execute(
            select(Document)
            .options(
                selectinload(Document.categories),
                selectinload(Document.tags),
            )
            .where(
                Document.document_id == document_id
            )
        )

        document = result.scalars().first()

        if document is None:
            raise ValueError("Document not found")

        content_hash, file_size = (
            await asyncio.to_thread(
                calculate_stream_hash,
                stream,
            )
        )

        version_result = await self.db.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.content_hash == content_hash,
            )
        )

        existing_version = (
            version_result.scalars().first()
        )

        if existing_version is not None:
            raise ValueError(
                "An identical document version already exists"
            )

        version = document.current_version + 1

        return await self._aupload_document_version(
            document=document,
            stream=stream,
            file_name=file_name,
            content_type=content_type,
            version=version,
            content_hash=content_hash,
            file_size=file_size,
            background_tasks=background_tasks,
        )

    # ============================================================
    # SYNCHRONOUS VERSION UPLOAD
    # ============================================================

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

        if isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "_upload_document_version() requires "
                "a synchronous Session."
            )

        document_version_id = uuid.uuid4()
        stored_blob_path: str | None = None

        blob_path = (
            f"documents/{document.document_id}/"
            f"v{version}/{file_name}"
        )

        try:
            stored_blob_path = (
                self.storage_provider.upload(
                    path=blob_path,
                    stream=stream,
                    content_type=content_type,
                )
            )

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

            document.current_version = version

            self.db.commit()

            categories = [
                category.name
                for category in document.categories
            ]

            tags = [
                tag.name
                for tag in document.tags
            ]

            self.document_ingestion_service.ingest(
                document=document,
                blob_path=stored_blob_path,
                file_name=file_name,
                document_version_id=document_version_id,
                categories=categories,
                tags=tags,
            )

            return DocumentUploadResponse(
                document_id=document.document_id,
                document_version_id=document_version_id,
                document_name=document.document_name,
                version=version,
                status=document.status,
            )

        except Exception:
            self.db.rollback()

            if stored_blob_path:
                try:
                    self.storage_provider.delete(
                        stored_blob_path
                    )
                except Exception:
                    pass

            raise

    # ============================================================
    # ASYNCHRONOUS VERSION UPLOAD
    # ============================================================

    async def _aupload_document_version(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str,
        content_type: str,
        version: int,
        content_hash: str,
        file_size: int,
        background_tasks: BackgroundTasks | None = None,
    ) -> DocumentUploadResponse:

        if not isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "_aupload_document_version() requires "
                "an AsyncSession."
            )

        document_version_id = uuid.uuid4()
        stored_blob_path: str | None = None

        blob_path = (
            f"documents/{document.document_id}/"
            f"v{version}/{file_name}"
        )

        try:
            stored_blob_path = (
                await self.storage_provider.aupload(
                    path=blob_path,
                    stream=stream,
                    content_type=content_type,
                )
            )

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

            document.current_version = version
            document.status = "QUEUED"

            await self.db.commit()

            categories = [
                category.name
                for category in document.categories
            ]

            tags = [
                tag.name
                for tag in document.tags
            ]

            if background_tasks is not None:

                background_tasks.add_task(
                    self._run_background_ingestion,
                    document=document,
                    blob_path=stored_blob_path,
                    file_name=file_name,
                    document_version_id=document_version_id,
                    categories=categories,
                    tags=tags,
                )

            else:

                await self.document_ingestion_service.aingest(
                    document=document,
                    blob_path=stored_blob_path,
                    file_name=file_name,
                    document_version_id=document_version_id,
                    categories=categories,
                    tags=tags,
                )

            return DocumentUploadResponse(
                document_id=document.document_id,
                document_version_id=document_version_id,
                document_name=document.document_name,
                version=version,
                status=document.status,
            )

        except Exception:
            await self.db.rollback()

            if stored_blob_path:
                try:
                    await self.storage_provider.adelete(
                        stored_blob_path
                    )
                except Exception:
                    pass

            raise

    # ============================================================
    # BACKGROUND INGESTION
    # ============================================================

    async def _run_background_ingestion(
        self,
        document: Document,
        blob_path: str,
        file_name: str,
        document_version_id: uuid.UUID,
        categories: list[str],
        tags: list[str],
    ) -> None:
        """
        Execute RAG ingestion after the HTTP response.

        The task intentionally does not use the database session
        for ingestion because the upload transaction has already
        been committed.
        """

        try:

            document.status = "PROCESSING"

            await self.document_ingestion_service.aingest(
                document=document,
                blob_path=blob_path,
                file_name=file_name,
                document_version_id=document_version_id,
                categories=categories,
                tags=tags,
            )

            document.status = "INDEXED"

        except Exception:

            document.status = "FAILED"

            raise

    # ============================================================
    # CATEGORIES
    # ============================================================

    def _add_categories(
        self,
        document: Document,
        categories: list[str] | None,
    ) -> None:

        if not categories:
            return

        if isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "_add_categories() requires a synchronous Session."
            )

        for category_name in categories:

            category = (
                self.db.query(Category)
                .filter(
                    Category.name == category_name
                )
                .first()
            )

            if category is None:
                category = Category(
                    name=category_name
                )
                self.db.add(category)

            document.categories.append(category)

    async def _aadd_categories(
        self,
        document: Document,
        categories: list[str] | None,
    ) -> None:

        if not categories:
            return

        if not isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "_aadd_categories() requires an AsyncSession."
            )

        for category_name in categories:

            result = await self.db.execute(
                select(Category).where(
                    Category.name == category_name
                )
            )

            category = result.scalars().first()

            if category is None:
                category = Category(
                    name=category_name
                )
                self.db.add(category)

            document.categories.append(category)

    # ============================================================
    # TAGS
    # ============================================================

    def _add_tags(
        self,
        document: Document,
        tags: list[str] | None,
    ) -> None:

        if not tags:
            return

        if isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "_add_tags() requires a synchronous Session."
            )

        for tag_name in tags:

            tag = (
                self.db.query(Tag)
                .filter(
                    Tag.name == tag_name
                )
                .first()
            )

            if tag is None:
                tag = Tag(
                    name=tag_name
                )
                self.db.add(tag)

            document.tags.append(tag)

    async def _aadd_tags(
        self,
        document: Document,
        tags: list[str] | None,
    ) -> None:

        if not tags:
            return

        if not isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "_aadd_tags() requires an AsyncSession."
            )

        for tag_name in tags:

            result = await self.db.execute(
                select(Tag).where(
                    Tag.name == tag_name
                )
            )

            tag = result.scalars().first()

            if tag is None:
                tag = Tag(
                    name=tag_name
                )
                self.db.add(tag)

            document.tags.append(tag)

    # ============================================================
    # ACTIVE DOCUMENTS
    # ============================================================

    def get_active_documents(
        self,
    ) -> DocumentListResponse:

        if isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "get_active_documents() requires a "
                "synchronous Session."
            )

        documents = (
            self.db.query(Document)
            .options(
                selectinload(Document.categories),
                selectinload(Document.tags),
            )
            .filter(
                Document.deleted_at.is_(None)
            )
            .order_by(
                Document.created_at.desc()
            )
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
                        for category
                        in document.categories
                    ],
                    tags=[
                        tag.name
                        for tag
                        in document.tags
                    ],
                    created_at=document.created_at,
                )
                for document in documents
            ]
        )

    async def aget_active_documents(
        self,
    ) -> DocumentListResponse:

        if not isinstance(self.db, AsyncSession):
            raise RuntimeError(
                "aget_active_documents() requires "
                "an AsyncSession."
            )

        result = await self.db.execute(
            select(Document)
            .options(
                selectinload(Document.categories),
                selectinload(Document.tags),
            )
            .where(
                Document.deleted_at.is_(None)
            )
            .order_by(
                Document.created_at.desc()
            )
        )

        documents = result.scalars().all()

        return DocumentListResponse(
            documents=[
                DocumentListItem(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    current_version=document.current_version,
                    status=document.status,
                    categories=[
                        category.name
                        for category
                        in document.categories
                    ],
                    tags=[
                        tag.name
                        for tag
                        in document.tags
                    ],
                    created_at=document.created_at,
                )
                for document in documents
            ]
        )