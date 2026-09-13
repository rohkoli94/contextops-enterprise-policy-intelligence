import pytest

from app.rag.workflow.nodes.cache_store import (
    create_cache_store_node,
)


class FakeCacheProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def set(
        self,
        key: str,
        value: dict,
        *,
        ttl_seconds: int,
    ) -> None:
        self.calls.append(
            {
                "key": key,
                "value": value,
                "ttl_seconds": ttl_seconds,
            }
        )

    async def get(self, key: str):
        return None

    async def delete(self, key: str) -> None:
        return None


class FailingCacheProvider(FakeCacheProvider):
    async def set(
        self,
        key: str,
        value: dict,
        *,
        ttl_seconds: int,
    ) -> None:
        raise RuntimeError(
            "Redis unavailable"
        )


# ============================================================
# SUCCESS
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_writes_grounded_response(
    monkeypatch,
) -> None:
    provider = FakeCacheProvider()

    monkeypatch.setattr(
        "app.rag.workflow.nodes.cache_store.settings.redis_cache_ttl_seconds",
        300,
    )

    node = create_cache_store_node(
        provider
    )

    state = {
        "cache_key": "contextops:query:test",
        "answer": (
            "The notice period is three months."
        ),
        "citations": [
            {
                "source": 1,
                "chunk_id": "chunk-001",
            }
        ],
        "grounding_status": "grounded",
    }

    result = await node(state)

    assert result["cache_write_attempted"] is True
    assert result["cache_written"] is True
    assert result["cache_write_error"] is None

    assert len(provider.calls) == 1

    assert provider.calls[0]["key"] == (
        "contextops:query:test"
    )

    assert provider.calls[0]["value"] == {
        "answer": (
            "The notice period is three months."
        ),
        "citations": [
            {
                "source": 1,
                "chunk_id": "chunk-001",
            }
        ],
    }

    assert provider.calls[0]["ttl_seconds"] == 300


# ============================================================
# MISSING CACHE KEY
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_skips_missing_cache_key() -> None:
    provider = FakeCacheProvider()

    node = create_cache_store_node(
        provider
    )

    result = await node(
        {
            "answer": "Some answer",
            "citations": [],
            "grounding_status": "grounded",
        }
    )

    assert result["cache_write_attempted"] is False
    assert result["cache_written"] is False
    assert result["cache_write_error"] is None
    assert provider.calls == []


# ============================================================
# EMPTY ANSWER
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_skips_empty_answer() -> None:
    provider = FakeCacheProvider()

    node = create_cache_store_node(
        provider
    )

    result = await node(
        {
            "cache_key": "cache-key",
            "answer": "   ",
            "citations": [],
            "grounding_status": "grounded",
        }
    )

    assert result["cache_write_attempted"] is False
    assert result["cache_written"] is False
    assert provider.calls == []


# ============================================================
# INVALID CITATIONS
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_skips_non_list_citations() -> None:
    provider = FakeCacheProvider()

    node = create_cache_store_node(
        provider
    )

    result = await node(
        {
            "cache_key": "cache-key",
            "answer": "Valid answer",
            "citations": "invalid",
            "grounding_status": "grounded",
        }
    )

    assert result["cache_write_attempted"] is False
    assert result["cache_written"] is False
    assert provider.calls == []


# ============================================================
# UNGROUNDED RESPONSE
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_skips_ungrounded_response() -> None:
    provider = FakeCacheProvider()

    node = create_cache_store_node(
        provider
    )

    result = await node(
        {
            "cache_key": "cache-key",
            "answer": "Potentially unsafe answer",
            "citations": [],
            "grounding_status": "not_grounded",
        }
    )

    assert result["cache_write_attempted"] is False
    assert result["cache_written"] is False
    assert provider.calls == []


# ============================================================
# REDIS FAILURE
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_does_not_fail_query_on_cache_error(
    monkeypatch,
) -> None:
    provider = FailingCacheProvider()

    monkeypatch.setattr(
        "app.rag.workflow.nodes.cache_store.settings.redis_cache_ttl_seconds",
        300,
    )

    node = create_cache_store_node(
        provider
    )

    result = await node(
        {
            "cache_key": "cache-key",
            "answer": "Valid grounded answer",
            "citations": [],
            "grounding_status": "grounded",
        }
    )

    assert result["cache_write_attempted"] is True
    assert result["cache_written"] is False
    assert result["cache_write_error"] == (
        "Redis unavailable"
    )


# ============================================================
# INVALID TTL
# ============================================================

@pytest.mark.asyncio
async def test_cache_store_rejects_non_positive_ttl(
    monkeypatch,
) -> None:
    provider = FakeCacheProvider()

    monkeypatch.setattr(
        "app.rag.workflow.nodes.cache_store.settings.redis_cache_ttl_seconds",
        0,
    )

    node = create_cache_store_node(
        provider
    )

    result = await node(
        {
            "cache_key": "cache-key",
            "answer": "Valid grounded answer",
            "citations": [],
            "grounding_status": "grounded",
        }
    )

    assert result["cache_write_attempted"] is False
    assert result["cache_written"] is False
    assert (
        result["cache_write_error"]
        == "redis_cache_ttl_seconds must be greater than zero."
    )

    assert provider.calls == []