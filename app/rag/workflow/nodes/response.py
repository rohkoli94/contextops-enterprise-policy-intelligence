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
        optional cache write-back

    have completed.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        grounding_status = state.get(
            "grounding_status"
        )

        # ----------------------------------------------------
        # FINAL RESPONSE METADATA
        # ----------------------------------------------------

        response_metadata: dict[str, Any] = {
            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            "retrieval_confidence": state.get(
                "retrieval_confidence"
            ),
            "retrieval_score": state.get(
                "retrieval_score"
            ),
            "retrieval_sufficient": state.get(
                "retrieval_sufficient"
            ),
            "retrieval_reason": state.get(
                "retrieval_reason"
            ),

            # ------------------------------------------------
            # Query rewriting
            # ------------------------------------------------

            "query_rewritten": state.get(
                "query_rewritten",
                False,
            ),
            "query_rewrite_fallback": state.get(
                "query_rewrite_fallback",
                False,
            ),

            # ------------------------------------------------
            # Cache
            # ------------------------------------------------

            "cache_hit": state.get(
                "cache_hit",
                False,
            ),
            "cache_written": state.get(
                "cache_written",
                False,
            ),
            "cache_write_error": state.get(
                "cache_write_error"
            ),

            # ------------------------------------------------
            # ContextOps
            # ------------------------------------------------

            "context_token_count": state.get(
                "context_token_count",
                0,
            ),
            "context_pii_detected": state.get(
                "context_pii_detected",
                False,
            ),
            "context_pii_entity_count": state.get(
                "context_pii_entity_count",
                0,
            ),
            "context_compressed": state.get(
                "context_compressed",
                False,
            ),
            "context_compressed_document_count": state.get(
                "context_compressed_document_count",
                0,
            ),

            # ------------------------------------------------
            # Grounding
            # ------------------------------------------------

            "grounding_status": grounding_status,
            "grounding_reason": state.get(
                "grounding_reason"
            ),

            # ------------------------------------------------
            # Observability
            # ------------------------------------------------
            #
            # Contains per-stage execution timings and the
            # total query execution time.
            #

            "timings": state.get(
                "timings",
                {
                    "stages": {},
                    "total_ms": None,
                },
            ),
        }

        # ----------------------------------------------------
        # FINAL STATE
        # ----------------------------------------------------

        return {
            **state,
            "final_response_metadata": response_metadata,
        }

    return node