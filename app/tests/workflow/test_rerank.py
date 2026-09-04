import pytest
from langchain_core.documents import Document

from app.rag.retrieval.pass_through_reranker import (
    PassThroughReranker,
)
from app.rag.workflow.nodes.rerank import (
    create_rerank_node,
)


class FakeReranker:
    async def rerank(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

        assert (
            query
            == "What is the notice period policy for managers?"
        )

        assert len(documents) == 2

        return list(reversed(documents))


@pytest.mark.asyncio
async def test_rerank_node_uses_contextualized_query() -> None:
    node = create_rerank_node(
        FakeReranker()
    )

    documents = [
        Document(
            page_content="Document A",
            metadata={"chunk_id": "a"},
        ),
        Document(
            page_content="Document B",
            metadata={"chunk_id": "b"},
        ),
    ]

    state = {
        "query": "What about managers?",
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "tenant_id": "tenant-001",
        "retrieved_documents": documents,
    }

    result = await node(state)

    assert result["reranked_documents"][0].page_content == (
        "Document B"
    )

    assert result["reranked_documents"][1].page_content == (
        "Document A"
    )

    # Original retrieval results remain unchanged.
    assert result["retrieved_documents"][0].page_content == (
        "Document A"
    )


@pytest.mark.asyncio
async def test_rerank_node_handles_empty_results() -> None:
    node = create_rerank_node(
        FakeReranker()
    )

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "retrieved_documents": [],
    }

    result = await node(state)

    assert result["reranked_documents"] == []


@pytest.mark.asyncio
async def test_pass_through_reranker_preserves_documents() -> None:
    reranker = PassThroughReranker()

    documents = [
        Document(
            page_content="Document A",
            metadata={"chunk_id": "a"},
        ),
        Document(
            page_content="Document B",
            metadata={"chunk_id": "b"},
        ),
    ]

    result = await reranker.rerank(
        query="What is the leave policy?",
        documents=documents,
    )

    assert len(result) == 2

    assert result[0].page_content == "Document A"
    assert result[1].page_content == "Document B"

    assert (
        result[0].metadata["reranker_applied"]
        is False
    )

    assert (
        result[0].metadata["reranker_rank"]
        == 1
    )

    assert (
        result[0].metadata["reranker_score"]
        is None
    )