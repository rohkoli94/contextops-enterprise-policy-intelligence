import pytest

from app.rag.retrieval.hybrid import (
    HybridRetriever,
)


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.received_query = None

    async def agenerate(
        self,
        request,
    ):
        self.received_query = request.text

        class Response:
            vector = [1.0, 0.0]

        return Response()


class FakeSparseEmbeddingProvider:
    async def agenerate(
        self,
        query,
    ):
        return {
            "indices": [1],
            "values": [1.0],
        }


class FakeVectorStore:
    async def asearch_hybrid(
        self,
        *,
        query_vector,
        sparse_query,
        tenant_id,
        top_k,
        filters,
    ):
        return []


@pytest.mark.asyncio
async def test_aretrieve_delegates_to_vector_store() -> None:
    embedding = FakeEmbeddingProvider()

    retriever = HybridRetriever(
        embedding_provider=embedding,
        sparse_embedding_provider=(
            FakeSparseEmbeddingProvider()
        ),
        vector_store=FakeVectorStore(),
    )

    result = await retriever.aretrieve(
        query="leave policy",
        tenant_id="tenant-001",
        top_k=5,
        filters=None,
    )

    assert result == []
    assert embedding.received_query == (
        "leave policy"
    )


def test_retriever_implements_abstract_contract() -> None:
    retriever = HybridRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedding_provider=(
            FakeSparseEmbeddingProvider()
        ),
        vector_store=FakeVectorStore(),
    )

    assert isinstance(
        retriever,
        HybridRetriever,
    )


def test_sync_retrieve_rejects_running_event_loop() -> None:
    async def invoke():
        retriever = HybridRetriever(
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedding_provider=(
                FakeSparseEmbeddingProvider()
            ),
            vector_store=FakeVectorStore(),
        )

        with pytest.raises(
            RuntimeError,
            match="running event loop",
        ):
            retriever.retrieve(
                query="leave policy",
                tenant_id="tenant-001",
                top_k=5,
            )

    import asyncio

    asyncio.run(
        invoke()
    )