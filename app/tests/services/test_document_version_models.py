import uuid

from app.models.document import Document
from app.models.document_version import DocumentVersion


def test_document_exposes_separate_current_and_active_version_fields() -> None:
    document = Document(
        document_name="policy.pdf",
        current_version=2,
        active_version_id=uuid.uuid4(),
        status="INDEXED",
    )

    assert document.current_version == 2
    assert document.active_version_id is not None


def test_document_version_default_status_is_pending() -> None:
    version = DocumentVersion(
        document_id=uuid.uuid4(),
        version=1,
        file_name="policy.pdf",
        content_type="application/pdf",
        file_size=10,
        content_hash="a" * 64,
        blob_path="documents/test/v1/policy.pdf",
    )

    assert version.status == "PENDING"
