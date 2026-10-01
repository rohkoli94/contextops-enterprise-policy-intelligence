import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Latest version that has been uploaded to Blob/PostgreSQL.
    # This is intentionally different from active_version_id.
    current_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # Version currently served by RAG retrieval.
    # A newly uploaded version must not become active until
    # extraction, chunking, embedding, Qdrant indexing and cache
    # invalidation have completed successfully.
    active_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_versions.document_version_id",
            name="fk_documents_active_version_id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        order_by="DocumentVersion.version",
    )

    active_version: Mapped["DocumentVersion | None"] = relationship(
        "DocumentVersion",
        foreign_keys=[active_version_id],
        post_update=True,
    )

    categories: Mapped[list["Category"]] = relationship(
        secondary="document_categories",
        back_populates="documents",
    )

    tags: Mapped[list["Tag"]] = relationship(
        secondary="document_tags",
        back_populates="documents",
    )
