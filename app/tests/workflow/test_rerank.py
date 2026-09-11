import pytest

from app.domain.document_chunk import DocumentChunk
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.pass_through_reranker import (
    PassThroughReranker,
)
from app.rag.workflow.nodes.rerank import (
    create_rerank_node,
)


def create_test_chunk(
    chunk_id: str,
    content: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        document_version_id="version-001",
        element_ids=[f"element-{chunk_id}"],
        content=content,
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
        metadata={},
    )


def create_test_retrieved_chunk(
    chunk_id: str,
    content: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=create_test_chunk(
            chunk_id,
            content,
        ),
        score=score,
        metadata={
            "chunk_id": chunk_id,
        },
    )


class FakeReranker:
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        assert (
            query
            == "What is the notice period policy for managers?"
        )

        assert len(candidates) == 2

        return list(reversed(candidates))


class CandidateCapturingReranker:
    def __init__(self) -> None:
        self.received_candidates: list[
            RetrievedChunk
        ] = []

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        self.received_candidates = list(candidates)

        return list(candidates)


@pytest.mark.asyncio
async def test_rerank_node_uses_contextualized_query() -> None:
    node = create_rerank_node(
        FakeReranker()
    )

    candidates = [
        create_test_retrieved_chunk(
            "a",
            "Document A",
            0.90,
        ),
        create_test_retrieved_chunk(
            "b",
            "Document B",
            0.80,
        ),
    ]

    state = {
        "query": "What about managers?",
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "tenant_id": "tenant-001",
        "retrieved_documents": candidates,
    }

    result = await node(state)

    assert (
        result["reranked_documents"][0].chunk.chunk_id
        == "b"
    )

    assert (
        result["reranked_documents"][1].chunk.chunk_id
        == "a"
    )

    # Original retrieval results remain unchanged.
    assert (
        result["retrieved_documents"][0].chunk.chunk_id
        == "a"
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
async def test_rerank_node_limits_candidate_pool() -> None:
    reranker = CandidateCapturingReranker()

    node = create_rerank_node(
        reranker,
        candidate_limit=2,
    )

    candidates = [
        create_test_retrieved_chunk(
            "a",
            "Document A",
            0.90,
        ),
        create_test_retrieved_chunk(
            "b",
            "Document B",
            0.80,
        ),
        create_test_retrieved_chunk(
            "c",
            "Document C",
            0.70,
        ),
        create_test_retrieved_chunk(
            "d",
            "Document D",
            0.60,
        ),
    ]

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "retrieved_documents": candidates,
    }

    result = await node(state)

    assert len(
        reranker.received_candidates
    ) == 2

    assert (
        reranker.received_candidates[0].chunk.chunk_id
        == "a"
    )

    assert (
        reranker.received_candidates[1].chunk.chunk_id
        == "b"
    )

    assert len(
        result["reranked_documents"]
    ) == 2

    # The original retrieval pool is preserved.
    assert len(
        result["retrieved_documents"]
    ) == 4


@pytest.mark.asyncio
async def test_rerank_node_rejects_invalid_candidate_limit() -> None:
    with pytest.raises(ValueError):
        create_rerank_node(
            FakeReranker(),
            candidate_limit=0,
        )


@pytest.mark.asyncio
async def test_pass_through_reranker_preserves_candidates() -> None:
    reranker = PassThroughReranker()

    candidates = [
        create_test_retrieved_chunk(
            "a",
            "Document A",
            0.90,
        ),
        create_test_retrieved_chunk(
            "b",
            "Document B",
            0.80,
        ),
    ]

    result = await reranker.rerank(
        query="What is the leave policy?",
        candidates=candidates,
    )

    assert len(result) == 2

    assert result[0].chunk.chunk_id == "a"
    assert result[1].chunk.chunk_id == "b"

    # Retrieval scores are preserved.
    assert result[0].score == 0.90
    assert result[1].score == 0.80

    # No real reranker has been applied yet.
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

    assert (
        result[0].reranker_score
        is None
    )