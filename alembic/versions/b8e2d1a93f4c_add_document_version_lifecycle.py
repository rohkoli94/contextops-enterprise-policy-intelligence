"""add document version lifecycle state"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b8e2d1a93f4c"
down_revision: Union[str, Sequence[str], None] = "0aeb2b280523"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default="PENDING",
        ),
    )
    op.create_index(
        "ix_document_versions_status",
        "document_versions",
        ["status"],
    )
    op.add_column(
        "documents",
        sa.Column("active_version_id", sa.UUID(), nullable=True),
    )
    op.create_index(
        "ix_documents_active_version_id",
        "documents",
        ["active_version_id"],
    )
    op.create_foreign_key(
        "fk_documents_active_version_id",
        "documents",
        "document_versions",
        ["active_version_id"],
        ["document_version_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_documents_active_version_id",
        "documents",
        type_="foreignkey",
    )
    op.drop_index(
        "ix_documents_active_version_id",
        table_name="documents",
    )
    op.drop_column("documents", "active_version_id")
    op.drop_index(
        "ix_document_versions_status",
        table_name="document_versions",
    )
    op.drop_column("document_versions", "status")
