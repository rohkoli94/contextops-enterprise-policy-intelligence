import pytest

from app.config.settings import settings
from app.domain.document_chunk import DocumentChunk
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_validator import (
    RetrievalEvaluation,
)
from app.rag.workflow.nodes.retrieval_recovery import (
    create_retrieval_recovery_node,
)


def make_chunk(
    *,
    chunk_id: str,
    content: str,
    score: float,
    reranker_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            document_version_id="version-001",
            element_ids=[f"element-{chunk_id}"],
            content=content,
            chunk_index=0,
            content_hash=f"hash-{chunk_id}",
            metadata={},
        ),
        score=score,
        metadata={
            "document_id": "doc-001",
            "document_version_id": "version-001",
            "chunk_id": chunk_id,
        },
        reranker_score=reranker_score,
    )


class FakeQueryRewriter:
    def __init__(
        self,
        rewritten_query: str = (
            "manager notice period policy "
            "employee resignation notice"
        ),
    ) -> None:
        self.rewritten_query = rewritten_query
        self.received_query: str | None = None
        self.received_summary: str | None = None
        self.received_messages: list[
            dict[str, object]
        ] | None = None

    async def rewrite(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        recent_messages: list[dict[str, object]],
    ) -> str:
        self.received_query = query
        self.received_summary = conversation_summary
        self.received_messages = recent_messages

        return self.rewritten_query


class FakeHybridRetriever:
    def __init__(
        self,
        documents: list[RetrievedChunk],
    ) -> None:
        self.documents = documents
        self.received_query: str | None = None
        self.received_tenant_id: str | None = None
        self.received_top_k: int | None = None
        self.received_filters: dict[str, object] | None = None

    async def aretrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        self.received_query = query
        self.received_tenant_id = tenant_id
        self.received_top_k = top_k
        self.received_filters = filters

        return self.documents


class FakeReranker:
    def __init__(
        self,
        documents: list[RetrievedChunk] | None = None,
    ) -> None:
        self.documents = documents
        self.received_query: str | None = None
        self.received_candidates: list[
            RetrievedChunk
        ] = []

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        self.received_query = query
        self.received_candidates = list(candidates)

        if self.documents is not None:
            return self.documents

        return list(candidates)


class FakeRetrievalValidator:
    def __init__(
        self,
        evaluation: RetrievalEvaluation,
    ) -> None:
        self.evaluation = evaluation
        self.received_query: str | None = None
        self.received_documents: list[
            RetrievedChunk
        ] = []

    async def evaluate(
        self,
        *,
        query: str,
        documents: list[RetrievedChunk],
    ) -> RetrievalEvaluation:
        self.received_query = query
        self.received_documents = list(documents)

        return self.evaluation


@pytest.mark.asyncio
async def test_recovery_reformulates_retrieves_reranks_and_validates() -> None:
    recovery_query = (
        "manager notice period policy "
        "employee resignation notice"
    )

    retrieved_documents = [
        make_chunk(
            chunk_id="chunk-001",
            content="Managers have a three-month notice period.",
            score=0.91,
            reranker_score=2.8,
        ),
        make_chunk(
            chunk_id="chunk-002",
            content="Employees must submit notice in writing.",
            score=0.84,
            reranker_score=1.1,
        ),
    ]

    reranked_documents = [
        make_chunk(
            chunk_id="chunk-001",
            content="Managers have a three-month notice period.",
            score=0.91,
            reranker_score=2.8,
        ),
        make_chunk(
            chunk_id="chunk-002",
            content="Employees must submit notice in writing.",
            score=0.84,
            reranker_score=1.1,
        ),
    ]

    query_rewriter = FakeQueryRewriter(
        rewritten_query=recovery_query,
    )

    hybrid_retriever = FakeHybridRetriever(
        retrieved_documents,
    )

    reranker = FakeReranker(
        documents=reranked_documents,
    )

    validator = FakeRetrievalValidator(
        RetrievalEvaluation(
            sufficient=True,
            confidence="strong",
            score=2.8,
            reason="Strong evidence found after recovery.",
            signals={
                "document_count": 2,
                "usable_document_count": 2,
            },
        )
    )

    node = create_retrieval_recovery_node(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        retrieval_validator=validator,
        query_rewriter=query_rewriter,
        max_retries=1,
    )

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "filters": {
            "content_type": "text",
        },
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "conversation_summary": (
            "User is asking about notice period policy."
        ),
        "recent_messages": [
            {
                "role": "user",
                "content": "What is the notice period?",
            }
        ],
        "retry_count": 0,
    }

    result = await node(state)

    # --------------------------------------------------------
    # Original query must remain unchanged.
    # --------------------------------------------------------

    assert result["query"] == "What about managers?"

    # --------------------------------------------------------
    # Recovery query is stored separately.
    # --------------------------------------------------------

    assert result["recovery_query"] == recovery_query

    # --------------------------------------------------------
    # Recovery succeeded.
    # --------------------------------------------------------

    assert result["recovery_success"] is True

    assert result["recovery_strategy"] == (
        "query_reformulation"
    )

    # --------------------------------------------------------
    # Retry count increments exactly once.
    # --------------------------------------------------------

    assert result["retry_count"] == 1

    # --------------------------------------------------------
    # Recovered documents are stored.
    # --------------------------------------------------------

    assert (
        result["retrieved_documents"]
        == retrieved_documents
    )

    assert (
        result["reranked_documents"]
        == reranked_documents
    )

    # --------------------------------------------------------
    # Validation result is propagated.
    # --------------------------------------------------------

    assert result["retrieval_sufficient"] is True

    assert (
        result["retrieval_confidence"]
        == "strong"
    )

    assert result["retrieval_score"] == 2.8

    assert result["retrieval_reason"] == (
        "Strong evidence found after recovery."
    )

    assert (
        result["retrieval_signals"][
            "document_count"
        ]
        == 2
    )

    # --------------------------------------------------------
    # Query rewriter receives the previous retrieval query.
    # --------------------------------------------------------

    assert query_rewriter.received_query is not None

    assert (
        "What is the notice period policy for managers?"
        in query_rewriter.received_query
    )

    assert (
        query_rewriter.received_summary
        == "User is asking about notice period policy."
    )

    # --------------------------------------------------------
    # Recovery retrieval uses reformulated query.
    # --------------------------------------------------------

    assert (
        hybrid_retriever.received_query
        == recovery_query
    )

    # --------------------------------------------------------
    # Tenant isolation is preserved.
    # --------------------------------------------------------

    assert (
        hybrid_retriever.received_tenant_id
        == "tenant-001"
    )

    # --------------------------------------------------------
    # Filters are preserved.
    # --------------------------------------------------------

    assert hybrid_retriever.received_filters == {
        "content_type": "text",
    }

    # --------------------------------------------------------
    # Recovery retrieves a broader pool.
    # --------------------------------------------------------

    assert (
        hybrid_retriever.received_top_k
        == max(
            settings.retrieval_top_k * 2,
            settings.rerank_candidate_limit,
        )
    )

    # --------------------------------------------------------
    # Re-ranking receives the recovery query.
    # --------------------------------------------------------

    assert (
        reranker.received_query
        == recovery_query
    )

    # --------------------------------------------------------
    # Candidate pool is bounded.
    # --------------------------------------------------------

    assert (
        len(reranker.received_candidates)
        <= settings.rerank_candidate_limit
    )

    assert (
        reranker.received_candidates
        == retrieved_documents[
            :settings.rerank_candidate_limit
        ]
    )

    # --------------------------------------------------------
    # Re-validation receives reranked documents.
    # --------------------------------------------------------

    assert (
        validator.received_query
        == recovery_query
    )

    assert (
        validator.received_documents
        == reranked_documents
    )


@pytest.mark.asyncio
async def test_recovery_safe_abstains_when_retry_limit_is_reached() -> None:
    query_rewriter = FakeQueryRewriter()

    hybrid_retriever = FakeHybridRetriever(
        documents=[],
    )

    reranker = FakeReranker()

    validator = FakeRetrievalValidator(
        RetrievalEvaluation(
            sufficient=False,
            confidence="none",
            score=None,
            reason="No evidence.",
            signals={
                "document_count": 0,
            },
        )
    )

    node = create_retrieval_recovery_node(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        retrieval_validator=validator,
        query_rewriter=query_rewriter,
        max_retries=1,
    )

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "filters": None,
        "contextualized_query": (
            "What is the leave policy?"
        ),
        "retry_count": 1,
    }

    result = await node(state)

    assert result["retry_count"] == 1

    assert result["recovery_strategy"] == (
        "safe_abstention"
    )

    assert result["recovery_success"] is False

    assert result["answer"] == (
        "I could not find sufficient policy evidence "
        "to answer this question reliably."
    )

    assert result["citations"] == []

    assert result["grounding_status"] == (
        "not_grounded"
    )

    assert result["grounding_supported_sources"] == []

    # No second recovery attempt.
    assert query_rewriter.received_query is None

    assert hybrid_retriever.received_query is None

    assert reranker.received_candidates == []

    assert validator.received_documents == []


@pytest.mark.asyncio
async def test_recovery_safe_abstains_when_retrieval_returns_nothing() -> None:
    query_rewriter = FakeQueryRewriter(
        rewritten_query=(
            "employee annual leave vacation policy"
        ),
    )

    hybrid_retriever = FakeHybridRetriever(
        documents=[],
    )

    reranker = FakeReranker()

    validator = FakeRetrievalValidator(
        RetrievalEvaluation(
            sufficient=False,
            confidence="none",
            score=None,
            reason="No evidence.",
            signals={},
        )
    )

    node = create_retrieval_recovery_node(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        retrieval_validator=validator,
        query_rewriter=query_rewriter,
        max_retries=1,
    )

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "filters": {
            "document_type": "policy",
        },
        "contextualized_query": (
            "What is the leave policy?"
        ),
        "retry_count": 0,
    }

    result = await node(state)

    assert result["retry_count"] == 1

    assert result["recovery_strategy"] == (
        "safe_abstention"
    )

    assert result["recovery_success"] is False

    assert result["recovery_query"] == (
        "employee annual leave vacation policy"
    )

    assert result["answer"] == (
        "I could not find sufficient policy evidence "
        "to answer this question reliably."
    )

    assert result["citations"] == []

    assert result["grounding_status"] == (
        "not_grounded"
    )

    assert result["grounding_supported_sources"] == []

    # Retrieval was attempted exactly once.
    assert (
        hybrid_retriever.received_query
        == "employee annual leave vacation policy"
    )

    assert (
        hybrid_retriever.received_tenant_id
        == "tenant-001"
    )

    assert hybrid_retriever.received_filters == {
        "document_type": "policy",
    }

    # No reranking or validation occurs when retrieval
    # returns no documents.
    assert reranker.received_candidates == []

    assert validator.received_documents == []