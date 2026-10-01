import pytest

from app.rag.workflow.nodes.cache_store import create_cache_store_node


class FakeCacheProvider:
    def __init__(self) -> None:
        self.calls = []

    async def set(self, key, value, *, ttl_seconds):
        self.calls.append((key, value, ttl_seconds))


@pytest.mark.asyncio
async def test_cache_store_persists_knowledge_base_version() -> None:
    provider = FakeCacheProvider()
    node = create_cache_store_node(provider)

    state = {
        "cache_key": "contextops:query:test",
        "knowledge_base_version": "9",
        "answer": "Policy answer",
        "citations": [{"document_id": "doc-1"}],
        "grounding_status": "grounded",
    }

    result = await node(state)

    assert result["cache_written"] is True
    assert provider.calls[0][1]["knowledge_base_version"] == "9"
