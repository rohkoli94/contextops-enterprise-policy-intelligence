from collections.abc import Awaitable, Callable

from app.rag.workflow.state import QueryState
from app.services.cache_key import build_query_cache_key
from app.services.cache_provider import CacheProvider


def create_cache_lookup_node(
    cache_provider: CacheProvider,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph cache lookup node.

    Cache is an optimization, not a correctness dependency.

    Cache failure therefore results in a miss and the graph
    continues with retrieval.
    """

    async def node(state: QueryState) -> QueryState:
        tenant_id = state["tenant_id"]

        retrieval_query = state.get(
            "contextualized_query"
        ) or state["query"]

        filters = state.get("filters")

        cache_key = build_query_cache_key(
            tenant_id=tenant_id,
            query=retrieval_query,
            filters=filters,
        )

        try:
            entry = await cache_provider.get(cache_key)

        except Exception as exc:
            return {
                **state,
                "cache_key": cache_key,
                "cache_hit": False,
                "cached_response": None,
                "cache_error": str(exc),
            }

        if entry is None:
            return {
                **state,
                "cache_key": cache_key,
                "cache_hit": False,
                "cached_response": None,
                "cache_error": None,
            }

        return {
            **state,
            "cache_key": cache_key,
            "cache_hit": True,
            "cached_response": entry.value,
            "cache_error": None,
        }

    return node