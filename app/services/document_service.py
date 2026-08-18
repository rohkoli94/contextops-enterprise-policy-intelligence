import uuid
from typing import BinaryIO

from sqlalchemy.orm import Session
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
        document = (
            self.db.query(Document)
            .filter(Document.document_id == document_id)
            .first()
        )

        if document is None:
            raise ValueError("Document not found")

        content_hash, file_size = calculate_stream_hash(stream)

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
        document_version_id = uuid.uuid4()
        stored_blob_path = None

        blob_path = (
            f"documents/{document.document_id}/"
            f"v{version}/{file_name}"
        )

        try:
            stored_blob_path = self.storage_provider.upload(
                path=blob_path,
                stream=stream,
                content_type=content_type,
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
                    self.storage_provider.delete(stored_blob_path)
                except Exception:
                    pass

            raise

    def _add_categories(
        self,
        document: Document,
        categories: list[str] | None,
    ) -> None:
        if not categories:
            return

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

    def _add_tags(
        self,
        document: Document,
        tags: list[str] | None,
    ) -> None:
        if not tags:
            return

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


def get_active_documents(self) -> DocumentListResponse:
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