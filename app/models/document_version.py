import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "version",
            name="uq_document_version",
        ),
        UniqueConstraint(
            "document_id",
            "content_hash",
            name="uq_document_content_hash",
        ),
        Index(
            "ix_document_versions_status",
            "status",
        ),
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("status", "PENDING")
        super().__init__(**kwargs)

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.document_id",
            name="fk_document_versions_document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    content_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    blob_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    # Lifecycle:
    #
    # PENDING -> PROCESSING -> ACTIVE
    #                       -> FAILED
    #
    # ACTIVE -> SUPERSEDED
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
        server_default="PENDING",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped["Document"] = relationship(
        back_populates="versions",
        foreign_keys=[document_id],
    )
