from collections.abc import Awaitable, Callable

from app.rag.workflow.state import QueryState
from app.services.query_rewriter import QueryRewriter


def create_query_contextualization_node(
    query_rewriter: QueryRewriter,
) -> Callable[
    [QueryState],
    Awaitable[QueryState],
]:
    """
    Create the LangGraph query-contextualization node.

    Query contextualization is only required when the user query
    depends on existing conversation context.

    For a genuinely standalone query, the query-rewriter LLM
    is skipped.

    Conversation context can be present through:

        - conversation_id
        - conversation_summary
        - recent_messages

    This allows the node to work correctly both in the real
    application flow and in isolated workflow tests.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        original_query = (
            state["query"].strip()
        )

        conversation_id = (
            state.get("conversation_id")
        )

        conversation_summary = (
            state.get("conversation_summary")
        )

        recent_messages = (
            state.get(
                "recent_messages",
                [],
            )
        )

        # ====================================================
        # DETERMINE WHETHER CONTEXT EXISTS
        # ====================================================

        has_conversation_context = (
            bool(conversation_id)
            or bool(conversation_summary)
            or bool(recent_messages)
        )

        # ====================================================
        # STANDALONE QUERY
        # ====================================================
        #
        # No conversation context means the original query is
        # already suitable for retrieval.
        #
        # IMPORTANT:
        # This avoids an unnecessary LLM call for first-time
        # standalone questions.
        # ====================================================

        if not has_conversation_context:
            return {
                **state,
                "contextualized_query": (
                    original_query
                ),
                "query_rewritten": False,
                "query_rewrite_fallback": False,
            }

        # ====================================================
        # CONTEXTUAL QUERY
        # ====================================================

        try:
            contextualized_query = (
                await query_rewriter.rewrite(
                    query=original_query,
                    conversation_summary=(
                        conversation_summary
                    ),
                    recent_messages=(
                        recent_messages
                    ),
                )
            )

            contextualized_query = (
                contextualized_query.strip()
            )

            # =================================================
            # EMPTY REWRITE FALLBACK
            # =================================================

            if not contextualized_query:
                return {
                    **state,
                    "contextualized_query": (
                        original_query
                    ),
                    "query_rewritten": False,
                    "query_rewrite_fallback": True,
                }

            # =================================================
            # SUCCESSFUL REWRITE
            # =================================================

            return {
                **state,
                "contextualized_query": (
                    contextualized_query
                ),
                "query_rewritten": (
                    contextualized_query
                    != original_query
                ),
                "query_rewrite_fallback": False,
            }

        # ====================================================
        # REWRITER FAILURE FALLBACK
        # ====================================================

        except Exception:
            return {
                **state,
                "contextualized_query": (
                    original_query
                ),
                "query_rewritten": False,
                "query_rewrite_fallback": True,
            }

    return node