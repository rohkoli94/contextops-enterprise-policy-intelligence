import pytest

from app.domain.document_chunk import DocumentChunk
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_validator import (
    RetrievalEvaluation,
)
from app.rag.workflow.nodes.retrieval_validation import (
    create_retrieval_validation_node,
)


class FakeRetrievalValidator:
    async def evaluate(
        self,
        *,
        query: str,
        documents: list[RetrievedChunk],
    ) -> RetrievalEvaluation:
        assert (
            query
            == "What is the notice period policy for managers?"
        )

        assert len(documents) == 2

        return RetrievalEvaluation(
            sufficient=True,
            confidence="strong",
            score=2.50,
            reason="Strong evidence found.",
            signals={
                "document_count": 2,
                "usable_document_count": 2,
                "reranker_score_count": 2,
                "top_reranker_score": 2.50,
                "second_reranker_score": 1.00,
                "reranker_score_gap": 1.50,
            },
        )


class FailingRetrievalValidator:
    async def evaluate(
        self,
        *,
        query: str,
        documents: list[RetrievedChunk],
    ) -> RetrievalEvaluation:
        raise RuntimeError(
            "Validator unavailable"
        )


def create_test_retrieved_chunk(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    retrieval_score: float,
    reranker_score: float | None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version_id=f"{document_id}-version",
            element_ids=[f"element-{chunk_id}"],
            content=content,
            chunk_index=0,
            content_hash=f"hash-{chunk_id}",
            metadata={},
        ),
        score=retrieval_score,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
        },
        reranker_score=reranker_score,
    )


@pytest.mark.asyncio
async def test_retrieval_validation_stores_evaluation() -> None:
    validator = FakeRetrievalValidator()

    node = create_retrieval_validation_node(
        validator
    )

    state = {
        "query": "What about managers?",
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "tenant_id": "tenant-001",
        "reranked_documents": [
            create_test_retrieved_chunk(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Manager policy",
                retrieval_score=0.91,
                reranker_score=2.50,
            ),
            create_test_retrieved_chunk(
                chunk_id="chunk-002",
                document_id="doc-002",
                content="Notice period policy",
                retrieval_score=0.82,
                reranker_score=1.00,
            ),
        ],
    }

    result = await node(state)

    assert result["retrieval_sufficient"] is True

    assert (
        result["retrieval_confidence"]
        == "strong"
    )

    assert result["retrieval_score"] == 2.50

    assert result["retrieval_reason"] == (
        "Strong evidence found."
    )

    assert (
        result["retrieval_signals"][
            "document_count"
        ]
        == 2
    )

    assert (
        result["retrieval_signals"][
            "reranker_score_gap"
        ]
        == 1.50
    )


@pytest.mark.asyncio
async def test_baseline_validator_rejects_empty_results() -> None:
    from app.rag.retrieval.baseline_retrieval_validator import (
        BaselineRetrievalValidator,
    )

    validator = BaselineRetrievalValidator()

    evaluation = await validator.evaluate(
        query="What is the leave policy?",
        documents=[],
    )

    assert evaluation.sufficient is False

    assert (
        evaluation.confidence
        == "none"
    )

    assert evaluation.score is None

    assert (
        evaluation.reason
        == "No documents were retrieved."
    )

    assert (
        evaluation.signals["document_count"]
        == 0
    )

    assert (
        evaluation.signals["usable_document_count"]
        == 0
    )

    assert (
        evaluation.signals["reranker_score_count"]
        == 0
    )


@pytest.mark.asyncio
async def test_baseline_validator_accepts_usable_documents() -> None:
    from app.rag.retrieval.baseline_retrieval_validator import (
        BaselineRetrievalValidator,
    )

    validator = BaselineRetrievalValidator()

    document = create_test_retrieved_chunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="Employees receive annual leave.",
        retrieval_score=0.80,
        reranker_score=2.50,
    )

    evaluation = await validator.evaluate(
        query="What is the leave policy?",
        documents=[document],
    )

    assert evaluation.sufficient is True

    assert (
        evaluation.confidence
        == "sufficient"
    )

    assert evaluation.score == 2.50

    assert (
        evaluation.reason
        == (
            "Usable reranked evidence is available, "
            "but confidence is not strongly separated."
        )
    )

    assert (
        evaluation.signals["document_count"]
        == 1
    )

    assert (
        evaluation.signals["usable_document_count"]
        == 1
    )

    assert (
        evaluation.signals["reranker_score_count"]
        == 1
    )

    assert (
        evaluation.signals["top_reranker_score"]
        == 2.50
    )

    assert (
        evaluation.signals["second_reranker_score"]
        is None
    )

    assert (
        evaluation.signals["reranker_score_gap"]
        is None
    )


@pytest.mark.asyncio
async def test_baseline_validator_classifies_strong_evidence() -> None:
    from app.rag.retrieval.baseline_retrieval_validator import (
        BaselineRetrievalValidator,
    )

    validator = BaselineRetrievalValidator()

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Managers have a three-month notice period.",
            retrieval_score=0.91,
            reranker_score=3.00,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            content="Employees must submit notice in writing.",
            retrieval_score=0.80,
            reranker_score=1.50,
        ),
    ]

    evaluation = await validator.evaluate(
        query="What is the notice period for managers?",
        documents=documents,
    )

    assert evaluation.sufficient is True

    assert (
        evaluation.confidence
        == "strong"
    )

    assert evaluation.score == 3.00

    assert (
        evaluation.signals[
            "reranker_score_gap"
        ]
        == 1.50
    )


@pytest.mark.asyncio
async def test_baseline_validator_handles_missing_reranker_scores() -> None:
    from app.rag.retrieval.baseline_retrieval_validator import (
        BaselineRetrievalValidator,
    )

    validator = BaselineRetrievalValidator()

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Employees receive annual leave.",
            retrieval_score=0.80,
            reranker_score=None,
        ),
    ]

    evaluation = await validator.evaluate(
        query="What is the leave policy?",
        documents=documents,
    )

    assert evaluation.sufficient is True

    assert (
        evaluation.confidence
        == "sufficient"
    )

    assert evaluation.score is None

    assert (
        evaluation.signals[
            "reranker_score_count"
        ]
        == 0
    )


@pytest.mark.asyncio
async def test_baseline_validator_rejects_documents_without_content() -> None:
    from app.rag.retrieval.baseline_retrieval_validator import (
        BaselineRetrievalValidator,
    )

    validator = BaselineRetrievalValidator()

    document = create_test_retrieved_chunk(
        chunk_id="chunk-empty",
        document_id="doc-empty",
        content="   ",
        retrieval_score=0.50,
        reranker_score=0.10,
    )

    evaluation = await validator.evaluate(
        query="What is the leave policy?",
        documents=[document],
    )

    assert evaluation.sufficient is False

    assert (
        evaluation.confidence
        == "none"
    )

    assert evaluation.score is None

    assert (
        evaluation.reason
        == (
            "Retrieved documents contain no usable content."
        )
    )

    assert (
        evaluation.signals["document_count"]
        == 1
    )

    assert (
        evaluation.signals["usable_document_count"]
        == 0
    )

    assert (
        evaluation.signals["reranker_score_count"]
        == 0
    )