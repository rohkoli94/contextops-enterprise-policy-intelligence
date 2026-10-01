from unittest.mock import AsyncMock, MagicMock

import pytest

from app.providers.vector_store.qdrant import QdrantVectorStore


def _store() -> QdrantVectorStore:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client = MagicMock()
    store.async_client = MagicMock()
    store.async_client.set_payload = AsyncMock()
    return store


def test_promote_document_version_updates_new_and_previous_version() -> None:
    store = _store()

    store.promote_document_version(
        document_id="doc-1",
        document_version_id="version-2",
        previous_active_version_id="version-1",
        tenant_id="contextops",
    )

    assert store.client.set_payload.call_count == 2
    calls = store.client.set_payload.call_args_list
    assert calls[0].kwargs["payload"] == {"version_status": "ACTIVE"}
    assert calls[1].kwargs["payload"] == {"version_status": "SUPERSEDED"}


@pytest.mark.asyncio
async def test_async_promote_document_version_updates_new_and_previous_version() -> None:
    store = _store()

    await store.apromote_document_version(
        document_id="doc-1",
        document_version_id="version-2",
        previous_active_version_id="version-1",
        tenant_id="contextops",
    )

    assert store.async_client.set_payload.await_count == 2
    calls = store.async_client.set_payload.await_args_list
    assert calls[0].kwargs["payload"] == {"version_status": "ACTIVE"}
    assert calls[1].kwargs["payload"] == {"version_status": "SUPERSEDED"}


def test_build_filter_requires_active_status() -> None:
    store = _store()

    query_filter = store._build_filter(
        tenant_id="tenant-001",
        filters=None,
    )

    assert query_filter.must[0].key == "tenant_id"
    assert query_filter.must[1].key == "version_status"
    assert query_filter.must[1].match.value == "ACTIVE"
