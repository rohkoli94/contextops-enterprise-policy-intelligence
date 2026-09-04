from collections.abc import Awaitable, Callable
from typing import Any

from app.rag.workflow.state import QueryState


def create_cached_response_node() -> (
    Callable[[QueryState], Awaitable[QueryState]]
):
    """
    Create the LangGraph cache-hit node.

    A cache hit bypasses:
        - retrieval
        - reranking
        - retrieval validation
        - context construction
        - LLM generation

    The cached response is copied into the normal answer/citation
    fields so the final Response node can use the same output path.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        cached_response = state.get(
            "cached_response"
        )

        if not cached_response:
            raise ValueError(
                "Cache-hit node requires cached_response."
            )

        answer = cached_response.get(
            "answer",
            "",
        )

        citations = cached_response.get(
            "citations",
            [],
        )

        if not isinstance(answer, str):
            raise ValueError(
                "Cached response answer must be a string."
            )

        if not isinstance(citations, list):
            raise ValueError(
                "Cached response citations must be a list."
            )

        final_metadata: dict[str, Any] = {
            **state.get(
                "final_response_metadata",
                {},
            ),
            "cache_hit": True,
            "response_source": "cache",
        }

        return {
            **state,
            "answer": answer,
            "citations": citations,
            "final_response_metadata": final_metadata,
        }

    return node