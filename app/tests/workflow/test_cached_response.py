import pytest

from app.rag.workflow.nodes.cached_response import (
    create_cached_response_node,
)


@pytest.mark.asyncio
async def test_cached_response_populates_answer_and_citations() -> None:
    node = create_cached_response_node()

    state = {
        "query": "What is the notice period?",
        "tenant_id": "tenant-001",
        "cache_hit": True,
        "cached_response": {
            "answer": (
                "The notice period is three months. [SOURCE 1]"
            ),
            "citations": [
                {
                    "source": 1,
                    "document_id": "doc-001",
                    "chunk_id": "chunk-001",
                }
            ],
        },
    }

    result = await node(state)

    assert result["answer"] == (
        "The notice period is three months. [SOURCE 1]"
    )

    assert len(result["citations"]) == 1

    assert (
        result["final_response_metadata"]["cache_hit"]
        is True
    )

    assert (
        result["final_response_metadata"]["response_source"]
        == "cache"
    )


@pytest.mark.asyncio
async def test_cached_response_requires_cached_payload() -> None:
    node = create_cached_response_node()

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "cache_hit": True,
        "cached_response": None,
    }

    with pytest.raises(
        ValueError,
        match="cached_response",
    ):
        await node(state)


@pytest.mark.asyncio
async def test_cached_response_rejects_invalid_answer() -> None:
    node = create_cached_response_node()

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "cache_hit": True,
        "cached_response": {
            "answer": 123,
            "citations": [],
        },
    }

    with pytest.raises(
        ValueError,
        match="answer must be a string",
    ):
        await node(state)