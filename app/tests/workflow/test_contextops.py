import pytest
from langchain_core.documents import Document

from app.rag.workflow.nodes.contextops import (
    create_contextops_node,
)


@pytest.mark.asyncio
async def test_contextops_builds_context_and_citations() -> None:
    node = create_contextops_node()

    documents = [
        Document(
            page_content=(
                "Managers have a three-month notice period."
            ),
            metadata={
                "document_id": "doc-001",
                "document_version_id": "version-001",
                "chunk_id": "chunk-001",
                "score": 0.91,
                "reranker_score": 0.88,
            },
        ),
        Document(
            page_content=(
                "Employees must submit notice in writing."
            ),
            metadata={
                "document_id": "doc-002",
                "document_version_id": "version-002",
                "chunk_id": "chunk-002",
                "score": 0.82,
                "reranker_score": 0.79,
            },
        ),
    ]

    state = {
        "query": "What is the notice period?",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert "[SOURCE 1]" in result["context"]
    assert "[SOURCE 2]" in result["context"]

    assert (
        "Managers have a three-month notice period."
        in result["context"]
    )

    assert len(result["citations"]) == 2

    assert (
        result["citations"][0]["document_id"]
        == "doc-001"
    )

    assert (
        result["citations"][0]["chunk_id"]
        == "chunk-001"
    )


@pytest.mark.asyncio
async def test_contextops_uses_configured_document_limit() -> None:
    node = create_contextops_node(
        max_documents=1,
    )

    documents = [
        Document(
            page_content="Document A",
            metadata={
                "document_id": "doc-a",
                "chunk_id": "chunk-a",
            },
        ),
        Document(
            page_content="Document B",
            metadata={
                "document_id": "doc-b",
                "chunk_id": "chunk-b",
            },
        ),
    ]

    state = {
        "query": "test",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert len(result["citations"]) == 1
    assert "Document A" in result["context"]
    assert "Document B" not in result["context"]


@pytest.mark.asyncio
async def test_contextops_handles_no_documents() -> None:
    node = create_contextops_node()

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "reranked_documents": [],
    }

    result = await node(state)

    assert (
        result["context"]
        == "No relevant policy documents were found."
    )

    assert result["citations"] == []


def test_contextops_rejects_invalid_document_limit() -> None:
    with pytest.raises(
        ValueError,
        match="context_max_documents",
    ):
        create_contextops_node(
            max_documents=0,
        )