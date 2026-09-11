import pytest

from app.domain.document_chunk import DocumentChunk
from app.rag.retrieval.models import RetrievedChunk
from app.rag.workflow.nodes.contextops import (
    create_contextops_node,
)


def create_test_retrieved_chunk(
    *,
    chunk_id: str,
    document_id: str,
    document_version_id: str,
    content: str,
    score: float | None,
    reranker_score: float | None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version_id=document_version_id,
            element_ids=[f"element-{chunk_id}"],
            content=content,
            chunk_index=0,
            content_hash=f"hash-{chunk_id}",
            metadata={},
        ),
        score=score if score is not None else 0.0,
        metadata={
            "document_id": document_id,
            "document_version_id": document_version_id,
            "chunk_id": chunk_id,
        },
        reranker_score=reranker_score,
    )


@pytest.mark.asyncio
async def test_contextops_builds_context_and_citations() -> None:
    node = create_contextops_node()

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            document_version_id="version-001",
            content=(
                "Managers have a three-month notice period."
            ),
            score=0.91,
            reranker_score=0.88,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            document_version_id="version-002",
            content=(
                "Employees must submit notice in writing."
            ),
            score=0.82,
            reranker_score=0.79,
        ),
    ]

    state = {
        "query": "What is the notice period?",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert (
        "Managers have a three-month notice period."
        in result["context"]
    )

    assert (
        "Employees must submit notice in writing."
        in result["context"]
    )

    assert "[SOURCE 1]" in result["context"]
    assert "[SOURCE 2]" in result["context"]

    assert len(result["citations"]) == 2

    assert result["citations"][0] == {
        "source": 1,
        "document_id": "doc-001",
        "document_version_id": "version-001",
        "chunk_id": "chunk-001",
        "score": 0.91,
        "reranker_score": 0.88,
    }

    assert result["citations"][1] == {
        "source": 2,
        "document_id": "doc-002",
        "document_version_id": "version-002",
        "chunk_id": "chunk-002",
        "score": 0.82,
        "reranker_score": 0.79,
    }


@pytest.mark.asyncio
async def test_contextops_uses_configured_document_limit() -> None:
    node = create_contextops_node(
        max_documents=1,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-a",
            document_id="doc-a",
            document_version_id="version-a",
            content="Document A",
            score=0.90,
            reranker_score=0.80,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-b",
            document_id="doc-b",
            document_version_id="version-b",
            content="Document B",
            score=0.80,
            reranker_score=0.70,
        ),
    ]

    state = {
        "query": "test",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert "Document A" in result["context"]
    assert "Document B" not in result["context"]

    assert len(result["citations"]) == 1
    assert (
        result["citations"][0]["document_id"]
        == "doc-a"
    )