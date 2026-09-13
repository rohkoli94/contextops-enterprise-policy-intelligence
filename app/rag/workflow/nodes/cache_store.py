from collections.abc import Awaitable, Callable
from typing import Any

from app.config.settings import settings
from app.rag.workflow.state import QueryState
from app.services.cache_provider import CacheProvider


def create_cache_store_node(
    cache_provider: CacheProvider,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph cache write-back node.

    Successful grounded responses are stored in the configured
    cache provider with a bounded TTL.

    Cache failures are deliberately non-fatal because caching is
    an optimization and must never become a correctness dependency.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:
        cache_key = state.get("cache_key")
        answer = state.get("answer")
        citations = state.get("citations", [])
        grounding_status = state.get("grounding_status")

        # ----------------------------------------------------
        # Default cache state
        # ----------------------------------------------------

        result_state: QueryState = {
            **state,
            "cache_write_attempted": False,
            "cache_written": False,
            "cache_write_error": None,
        }

        # ----------------------------------------------------
        # Cache only valid grounded responses
        # ----------------------------------------------------

        if not cache_key:
            return result_state

        if not isinstance(answer, str) or not answer.strip():
            return result_state

        if not isinstance(citations, list):
            return result_state

        if grounding_status != "grounded":
            return result_state

        # ----------------------------------------------------
        # Validate TTL
        # ----------------------------------------------------

        ttl_seconds = settings.redis_cache_ttl_seconds

        if ttl_seconds <= 0:
            return {
                **result_state,
                "cache_write_error": (
                    "redis_cache_ttl_seconds must be greater than zero."
                ),
            }

        # ----------------------------------------------------
        # Build cache payload
        # ----------------------------------------------------

        cache_payload: dict[str, Any] = {
            "answer": answer,
            "citations": citations,
        }

        # ----------------------------------------------------
        # Write cache
        # ----------------------------------------------------

        result_state["cache_write_attempted"] = True

        try:
            await cache_provider.set(
                cache_key,
                cache_payload,
                ttl_seconds=ttl_seconds,
            )

            result_state["cache_written"] = True

        except Exception as exc:
            result_state["cache_write_error"] = str(exc)

        return result_state

    return node