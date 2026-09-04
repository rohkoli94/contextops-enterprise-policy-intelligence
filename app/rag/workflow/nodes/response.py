from collections.abc import Awaitable, Callable
from typing import Any

from app.rag.workflow.state import QueryState


def create_response_node() -> (
    Callable[[QueryState], Awaitable[QueryState]]
):
    """
    Create the final response node.

    The node does not generate or validate the answer.

    It only assembles the final response metadata after:
        retrieval
        context construction
        LLM generation
        grounding validation
    have completed.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        grounding_status = state.get(
            "grounding_status"
        )

        response_metadata: dict[str, Any] = {
            "retrieval_confidence": state.get(
                "retrieval_confidence"
            ),
            "retrieval_score": state.get(
                "retrieval_score"
            ),
            "retrieval_sufficient": state.get(
                "retrieval_sufficient"
            ),
            "query_rewritten": state.get(
                "query_rewritten",
                False,
            ),
            "query_rewrite_fallback": state.get(
                "query_rewrite_fallback",
                False,
            ),
            "cache_hit": state.get(
                "cache_hit",
                False,
            ),
            "grounding_status": grounding_status,
            "grounding_reason": state.get(
                "grounding_reason"
            ),
        }

        return {
            **state,
            "final_response_metadata": response_metadata,
        }

    return node