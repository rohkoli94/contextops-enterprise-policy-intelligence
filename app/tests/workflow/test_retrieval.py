import pytest

from app.rag.workflow.nodes.retrieval import (
    create_hybrid_retrieval_node,
)
from app.rag.retrieval.models import RetrievedChunk


class FakeHybridRetriever:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_tenant_id: str | None = None
        self.received_top_k: int | None = None
        self.received_filters: dict | None = None

    async def aretrieve(
        self,
        *,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:

        self.received_query = query
        self.received_tenant_id = tenant_id
        self.received_top_k = top_k
        self.received_filters = filters

        return []


@pytest.mark.asyncio
async def test_retrieval_uses_contextualized_query() -> None:
    retriever = FakeHybridRetriever()

    node = create_hybrid_retrieval_node(retriever)

    state = {
        "query": "What about managers?",
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "tenant_id": "tenant-001",
        "filters": {
            "content_type": "text",
        },
    }

    result = await node(state)

    assert (
        retriever.received_query
        == "What is the notice period policy for managers?"
    )

    assert (
        retriever.received_query
        != state["query"]
    )

    assert (
        retriever.received_tenant_id
        == "tenant-001"
    )

    assert (
        retriever.received_filters
        == {"content_type": "text"}
    )

    assert result["retrieved_documents"] == []


@pytest.mark.asyncio
async def test_retrieval_falls_back_to_original_query() -> None:
    retriever = FakeHybridRetriever()

    node = create_hybrid_retrieval_node(retriever)

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "filters": None,
    }

    await node(state)

    assert (
        retriever.received_query
        == "What is the leave policy?"
    )