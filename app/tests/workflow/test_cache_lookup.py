import pytest

from app.rag.workflow.nodes.cache_lookup import (
    create_cache_lookup_node,
)
from app.services.cache_provider import CacheEntry


class FakeCacheProvider:
    def __init__(
        self,
        entry: CacheEntry | None = None,
    ) -> None:
        self.entry = entry
        self.requested_key: str | None = None

    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        self.requested_key = key
        return self.entry


class FailingCacheProvider:
    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        raise RuntimeError("Cache unavailable")


@pytest.mark.asyncio
async def test_cache_hit() -> None:
    provider = FakeCacheProvider(
        CacheEntry(
            value={
                "answer": "The notice period is three months.",
                "citations": [],
            },
            cache_key="existing-key",
        )
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

    assert result["cache_key"] is not None
    assert result["cache_key"].startswith(
        "contextops:query:"
    )


@pytest.mark.asyncio
async def test_cache_miss() -> None:
    provider = FakeCacheProvider()

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
    assert result["cache_error"] == "Cache unavailable"