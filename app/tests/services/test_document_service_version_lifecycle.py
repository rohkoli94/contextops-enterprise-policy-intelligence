import uuid
from types import SimpleNamespace
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.services.document_service import DocumentService


class _Result:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        return self.value


class _DB:
    def __init__(self, document, document_version, previous_version):
        self.document = document
        self.document_version = document_version
        self.previous_version = previous_version
        self.calls = 0
        self.commit_count = 0

    async def execute(self, statement):
        self.calls += 1
        if self.calls == 1:
            return _Result(self.document)
        if self.calls == 2:
            return _Result(self.document_version)
        return _Result(self.previous_version)

    async def commit(self):
        self.commit_count += 1

    async def rollback(self):
        return None


class _SessionContext:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_background_ingestion_promotes_active_version(monkeypatch):
    document_id = uuid.uuid4()
    version1_id = uuid.uuid4()
    version2_id = uuid.uuid4()

    document = Document(
        document_id=document_id,
        document_name="policy.pdf",
        current_version=2,
        active_version_id=version1_id,
        status="QUEUED",
    )
    version2 = DocumentVersion(
        document_version_id=version2_id,
        document_id=document_id,
        version=2,
        file_name="policy-v2.pdf",
        content_type="application/pdf",
        file_size=10,
        content_hash="b" * 64,
        blob_path="documents/test/v2/policy.pdf",
        status="PENDING",
    )
    previous = DocumentVersion(
        document_version_id=version1_id,
        document_id=document_id,
        version=1,
        file_name="policy-v1.pdf",
        content_type="application/pdf",
        file_size=10,
        content_hash="a" * 64,
        blob_path="documents/test/v1/policy.pdf",
        status="ACTIVE",
    )

    db = _DB(document, version2, previous)
    ingestion = SimpleNamespace(
        aingest=AsyncMock(return_value=[object()]),
        apromote_document_version=AsyncMock(),
    )
    service = object.__new__(DocumentService)
    service.document_ingestion_service = ingestion

    monkeypatch.setattr(
        "app.services.document_service.AsyncSessionLocal",
        lambda: _SessionContext(db),
    )
    monkeypatch.setattr(
        DocumentService,
        "_bump_knowledge_base_cache_version",
        AsyncMock(return_value="2"),
    )

    await service._run_background_ingestion(
        document_id=document_id,
        blob_path="documents/test/v2/policy.pdf",
        file_name="policy-v2.pdf",
        document_version_id=version2_id,
        version_number=2,
        categories=[],
        tags=[],
    )

    assert version2.status == "ACTIVE"
    assert previous.status == "SUPERSEDED"
    assert document.active_version_id == version2_id
    assert document.status == "INDEXED"
    ingestion.apromote_document_version.assert_awaited_once()
    assert ingestion.apromote_document_version.await_args.kwargs["document_version_id"] == version2_id
    assert ingestion.apromote_document_version.await_args.kwargs["previous_active_version_id"] == version1_id
class _Query:
    def __init__(self, value=None):
        self.value = value

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.value


class _SyncDB:
    def __init__(self):
        self.added = []
        self.commit_count = 0
        self.rollback_count = 0

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1

    def query(self, model):
        return _Query(None)


def test_synchronous_upload_promotes_active_version(monkeypatch):
    document = Document(
        document_id=uuid.uuid4(),
        document_name="policy.pdf",
        current_version=0,
        active_version_id=None,
        status="QUEUED",
    )
    db = _SyncDB()
    ingestion = SimpleNamespace(
        ingest=MagicMock(return_value=[object()]),
        promote_document_version=MagicMock(),
    )
    storage = SimpleNamespace(
        upload=MagicMock(return_value="documents/test/v1/policy.pdf"),
    )

    service = object.__new__(DocumentService)
    service.db = db
    service.storage_provider = storage
    service.document_ingestion_service = ingestion

    monkeypatch.setattr(
        DocumentService,
        "_bump_knowledge_base_cache_version",
        AsyncMock(return_value="1"),
    )

    result = service._upload_document_version(
        document=document,
        stream=BytesIO(b"policy"),
        file_name="policy.pdf",
        content_type="application/pdf",
        version=1,
        content_hash="a" * 64,
        file_size=6,
    )

    assert result.active_version == 1
    assert document.active_version_id is not None
    assert document.status == "INDEXED"
    assert len(db.added) == 1
    assert db.added[0].status == "ACTIVE"
    assert ingestion.promote_document_version.call_count == 1
