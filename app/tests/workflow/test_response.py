import pytest

from app.rag.workflow.nodes.response import (
    create_response_node,
)


@pytest.mark.asyncio
async def test_response_node_builds_final_metadata() -> None:
    node = create_response_node()

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "answer": (
            "The notice period is three months. [SOURCE 1]"
        ),
        "citations": [
            {
                "source": 1,
                "document_id": "doc-001",
                "document_version_id": "version-001",
                "chunk_id": "chunk-001",
            }
        ],
        "retrieval_confidence": "medium",
        "retrieval_score": None,
        "retrieval_sufficient": True,
        "query_rewritten": True,
        "query_rewrite_fallback": False,
        "cache_hit": False,
        "grounding_status": "grounded",
        "grounding_reason": (
            "Answer references available evidence sources."
        ),
    }

    result = await node(state)

    metadata = result["final_response_metadata"]

    assert metadata["retrieval_confidence"] == "medium"
    assert metadata["retrieval_sufficient"] is True
    assert metadata["query_rewritten"] is True
    assert metadata["query_rewrite_fallback"] is False
    assert metadata["cache_hit"] is False
    assert metadata["grounding_status"] == "grounded"

    # Original answer and citations remain intact.
    assert result["answer"].startswith(
        "The notice period is three months."
    )

    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_response_node_handles_missing_optional_metadata() -> None:
    node = create_response_node()

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "answer": "The policy is described in the evidence.",
    }

    result = await node(state)

    metadata = result["final_response_metadata"]

    assert metadata["query_rewritten"] is False
    assert metadata["query_rewrite_fallback"] is False
    assert metadata["cache_hit"] is False