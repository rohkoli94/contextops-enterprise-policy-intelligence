import pytest

from langchain_core.documents import Document

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
        documents: list[Document],
    ) -> RetrievalEvaluation:

        assert (
            query
            == "What is the notice period policy for managers?"
        )

        assert len(documents) == 2

        return RetrievalEvaluation(
            sufficient=True,
            confidence="high",
            score=0.91,
            reason="Strong evidence found.",
            signals={
                "document_count": 2,
            },
        )


class FailingRetrievalValidator:
    async def evaluate(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> RetrievalEvaluation:

        raise RuntimeError(
            "Validator unavailable"
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
            Document(
                page_content="Manager policy",
                metadata={"chunk_id": "1"},
            ),
            Document(
                page_content="Notice period policy",
                metadata={"chunk_id": "2"},
            ),
        ],
    }

    result = await node(state)

    assert result["retrieval_sufficient"] is True
    assert result["retrieval_confidence"] == "high"
    assert result["retrieval_score"] == 0.91
    assert result["retrieval_reason"] == (
        "Strong evidence found."
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
    assert evaluation.confidence == "low"


@pytest.mark.asyncio
async def test_baseline_validator_accepts_usable_documents() -> None:
    from app.rag.retrieval.baseline_retrieval_validator import (
        BaselineRetrievalValidator,
    )

    validator = BaselineRetrievalValidator()

    evaluation = await validator.evaluate(
        query="What is the leave policy?",
        documents=[
            Document(
                page_content="Employees receive annual leave.",
                metadata={},
            )
        ],
    )

    assert evaluation.sufficient is True
    assert evaluation.confidence == "medium"