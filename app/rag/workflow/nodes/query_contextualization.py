from collections.abc import Awaitable, Callable

from app.rag.workflow.state import QueryState
from app.services.query_rewriter import QueryRewriter


def create_query_contextualization_node(
    query_rewriter: QueryRewriter,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph query-contextualization node.
    """

    async def node(state: QueryState) -> QueryState:
        original_query = state["query"]

        try:
            contextualized_query = await query_rewriter.rewrite(
                query=original_query,
                conversation_summary=state.get(
                    "conversation_summary"
                ),
                recent_messages=state.get(
                    "recent_messages",
                    [],
                ),
            )

            contextualized_query = contextualized_query.strip()

            if not contextualized_query:
                contextualized_query = original_query.strip()

                return {
                    **state,
                    "contextualized_query": contextualized_query,
                    "query_rewritten": False,
                    "query_rewrite_fallback": True,
                }

            return {
                **state,
                "contextualized_query": contextualized_query,
                "query_rewritten": (
                    contextualized_query != original_query.strip()
                ),
                "query_rewrite_fallback": False,
            }

        except Exception:
            return {
                **state,
                "contextualized_query": original_query.strip(),
                "query_rewritten": False,
                "query_rewrite_fallback": True,
            }

    return node