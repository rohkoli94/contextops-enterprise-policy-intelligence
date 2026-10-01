import pytest

from app.rag.workflow.nodes.cache_lookup import (
    create_cache_lookup_node,
)
from app.services.cache_provider import CacheEntry


class FakeCacheProvider:
    def __init__(
        self,
        entry: CacheEntry | None = None,
        knowledge_base_version: str = "0",
    ) -> None:
        self.entry = entry
        self.knowledge_base_version = knowledge_base_version
        self.requested_key: str | None = None
        self.requested_tenant_id: str | None = None

    async def get_knowledge_base_version(
        self,
        tenant_id: str,
    ) -> str:
        self.requested_tenant_id = tenant_id
        return self.knowledge_base_version

    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        self.requested_key = key
        return self.entry


class FailingCacheProvider:
    async def get_knowledge_base_version(
        self,
        tenant_id: str,
    ) -> str:
        raise RuntimeError("Cache unavailable")

    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        raise AssertionError(
            "Query cache get must not run when the version lookup fails."
        )


@pytest.mark.asyncio
async def test_cache_hit() -> None:
    provider = FakeCacheProvider(
        CacheEntry(
            value={
                "answer": "The notice period is three months.",
                "citations": [],
            },
            cache_key="existing-key",
        ),
        knowledge_base_version="7",
    )

    node = create_cache_lookup_node(provider)

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "filters": None,
    }

    result = await node(state)

    assert result["cache_hit"] is True
    assert result["cached_response"] == {
        "answer": "The notice period is three months.",
        "citations": [],
    }
    assert result["knowledge_base_version"] == "7"
    assert provider.requested_tenant_id == "tenant-001"

    assert result["cache_key"] is not None
    assert result["cache_key"].startswith(
        "contextops:query:"
    )


@pytest.mark.asyncio
async def test_cache_miss() -> None:
    provider = FakeCacheProvider(
        knowledge_base_version="2"
    )

    node = create_cache_lookup_node(provider)

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "contextualized_query": (
            "What is the leave policy?"
        ),
        "filters": None,
    }

    result = await node(state)

    assert result["cache_hit"] is False
    assert result["cached_response"] is None
    assert result["cache_error"] is None
    assert result["knowledge_base_version"] == "2"


@pytest.mark.asyncio
async def test_cache_key_changes_when_knowledge_base_version_changes() -> None:
    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "contextualized_query": (
            "What is the leave policy?"
        ),
        "filters": None,
    }

    provider_v1 = FakeCacheProvider(
        knowledge_base_version="1"
    )
    provider_v2 = FakeCacheProvider(
        knowledge_base_version="2"
    )

    result_v1 = await create_cache_lookup_node(
        provider_v1
    )(state)
    result_v2 = await create_cache_lookup_node(
        provider_v2
    )(state)

    assert result_v1["cache_key"] != result_v2["cache_key"]


@pytest.mark.asyncio
async def test_cache_failure_does_not_fail_request() -> None:
    provider = FailingCacheProvider()

    node = create_cache_lookup_node(provider)

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "contextualized_query": (
            "What is the leave policy?"
        ),
        "filters": None,
    }

    result = await node(state)

    assert result["cache_hit"] is False
    assert result["cached_response"] is None
    assert result["cache_key"] is None
    assert result["knowledge_base_version"] is None
    assert result["cache_error"] == "Cache unavailable"


@pytest.mark.asyncio
async def test_stale_cache_entry_is_ignored() -> None:
    provider = FakeCacheProvider(
        CacheEntry(
            value={
                "answer": "old",
                "citations": [],
                "knowledge_base_version": "6",
            },
            cache_key="existing-key",
        ),
        knowledge_base_version="7",
    )

    node = create_cache_lookup_node(provider)

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "contextualized_query": "What is the leave policy?",
        "filters": None,
    }

    result = await node(state)

    assert result["cache_hit"] is False
    assert result["cached_response"] is None
    assert result["knowledge_base_version"] == "7"
