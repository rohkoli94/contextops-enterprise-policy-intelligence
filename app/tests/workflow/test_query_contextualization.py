import pytest

from app.rag.workflow.nodes.query_contextualization import (
    create_query_contextualization_node,
)


class FakeQueryRewriter:
    async def rewrite(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        recent_messages: list[dict[str, object]],
    ) -> str:
        assert query == "What about managers?"

        assert conversation_summary == (
            "User is asking about employee notice policies."
        )

        assert len(recent_messages) == 2

        return "What is the notice period policy for managers?"


class FailingQueryRewriter:
    async def rewrite(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        recent_messages: list[dict[str, object]],
    ) -> str:
        raise RuntimeError("Query rewriter unavailable")


@pytest.mark.asyncio
async def test_contextualizes_conversational_query() -> None:
    node = create_query_contextualization_node(
        FakeQueryRewriter()
    )

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "conversation_summary": (
            "User is asking about employee notice policies."
        ),
        "recent_messages": [
            {
                "role": "user",
                "content": "What is the notice period?",
            },
            {
                "role": "assistant",
                "content": "The policy states three months.",
            },
        ],
    }

    result = await node(state)

    # --------------------------------------------------
    # Original user query must remain unchanged.
    # --------------------------------------------------

    assert result["query"] == "What about managers?"

    # --------------------------------------------------
    # Contextualized query should be stored separately.
    # This query will later be used for retrieval.
    # --------------------------------------------------

    assert (
        result["contextualized_query"]
        == "What is the notice period policy for managers?"
    )

    # --------------------------------------------------
    # Observability metadata
    # --------------------------------------------------

    assert result["query_rewritten"] is True
    assert result["query_rewrite_fallback"] is False


@pytest.mark.asyncio
async def test_falls_back_to_original_query_when_rewriter_fails() -> None:
    node = create_query_contextualization_node(
        FailingQueryRewriter()
    )

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "conversation_summary": None,
        "recent_messages": [],
    }

    result = await node(state)

    # --------------------------------------------------
    # Original query must be used as retrieval query
    # when the contextualizer fails.
    # --------------------------------------------------

    assert (
        result["contextualized_query"]
        == "What about managers?"
    )

    # --------------------------------------------------
    # Observability metadata
    # --------------------------------------------------

    assert result["query_rewritten"] is False
    assert result["query_rewrite_fallback"] is True


@pytest.mark.asyncio
async def test_falls_back_when_rewriter_returns_empty_query() -> None:
    class EmptyQueryRewriter:
        async def rewrite(
            self,
            *,
            query: str,
            conversation_summary: str | None,
            recent_messages: list[dict[str, object]],
        ) -> str:
            return ""

    node = create_query_contextualization_node(
        EmptyQueryRewriter()
    )

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "conversation_summary": None,
        "recent_messages": [],
    }

    result = await node(state)

    assert (
        result["contextualized_query"]
        == "What about managers?"
    )

    assert result["query_rewritten"] is False
    assert result["query_rewrite_fallback"] is True