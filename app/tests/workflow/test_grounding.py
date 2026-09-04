import pytest

from langchain_core.documents import Document

from app.rag.retrieval.baseline_grounding_validator import (
    BaselineGroundingValidator,
)
from app.rag.workflow.nodes.grounding import (
    create_grounding_validation_node,
)


@pytest.mark.asyncio
async def test_grounding_accepts_answer_with_source_reference() -> None:
    validator = BaselineGroundingValidator()

    evaluation = await validator.evaluate(
        query="What is the notice period?",
        answer=(
            "The notice period is three months. [SOURCE 1]"
        ),
        documents=[
            Document(
                page_content=(
                    "The notice period is three months."
                ),
                metadata={
                    "chunk_id": "chunk-1",
                },
            )
        ],
    )

    assert evaluation.grounded is True
    assert evaluation.confidence == "medium"
    assert evaluation.supported_sources == [1]


@pytest.mark.asyncio
async def test_grounding_rejects_answer_without_evidence_reference() -> None:
    validator = BaselineGroundingValidator()

    evaluation = await validator.evaluate(
        query="What is the notice period?",
        answer="The notice period is three months.",
        documents=[
            Document(
                page_content=(
                    "The notice period is three months."
                ),
                metadata={
                    "chunk_id": "chunk-1",
                },
            )
        ],
    )

    assert evaluation.grounded is False
    assert evaluation.confidence == "low"
    assert evaluation.supported_sources == []


@pytest.mark.asyncio
async def test_grounding_rejects_missing_evidence() -> None:
    validator = BaselineGroundingValidator()

    evaluation = await validator.evaluate(
        query="What is the notice period?",
        answer="The notice period is three months.",
        documents=[],
    )

    assert evaluation.grounded is False
    assert evaluation.supported_sources == []


@pytest.mark.asyncio
async def test_grounding_node_updates_state() -> None:
    validator = BaselineGroundingValidator()

    node = create_grounding_validation_node(
        validator
    )

    state = {
        "query": "What is the notice period?",
        "tenant_id": "tenant-001",
        "answer": (
            "The notice period is three months. [SOURCE 1]"
        ),
        "reranked_documents": [
            Document(
                page_content=(
                    "The notice period is three months."
                ),
                metadata={
                    "chunk_id": "chunk-1",
                },
            )
        ],
    }

    result = await node(state)

    assert result["grounding_status"] == "grounded"
    assert result["grounding_supported_sources"] == [1]