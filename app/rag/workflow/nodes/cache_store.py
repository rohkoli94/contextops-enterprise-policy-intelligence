from collections.abc import Awaitable, Callable
from typing import Any

from app.config.settings import settings
from app.core.logging import get_logger
from app.rag.workflow.state import QueryState
from app.services.cache_provider import CacheProvider


logger = get_logger(__name__)


def create_cache_store_node(
    cache_provider: CacheProvider,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """Create the version-aware query cache write-back node."""

    async def node(state: QueryState) -> QueryState:
        cache_key = state.get("cache_key")
        answer = state.get("answer")
        citations = state.get("citations", [])
        grounding_status = state.get("grounding_status")
        knowledge_base_version = state.get("knowledge_base_version")

        result_state: QueryState = {
            **state,
            "cache_write_attempted": False,
            "cache_written": False,
            "cache_write_error": None,
        }

        if not cache_key:
            return result_state

        if not isinstance(answer, str) or not answer.strip():
            return result_state

        if not isinstance(citations, list):
            return result_state

        if grounding_status != "grounded":
            return result_state

        ttl_seconds = settings.redis_cache_ttl_seconds
        if ttl_seconds <= 0:
            error = "redis_cache_ttl_seconds must be greater than zero."
            logger.error(
                "Skipping query cache write because TTL is invalid",
                extra={"ttl_seconds": ttl_seconds},
            )
            return {
                **result_state,
                "cache_write_error": error,
            }

        cache_payload: dict[str, Any] = {
            "answer": answer,
            "citations": citations,
        }

        if knowledge_base_version is not None:
            cache_payload["knowledge_base_version"] = (
                knowledge_base_version
            )

        result_state["cache_write_attempted"] = True

        try:
            await cache_provider.set(
                cache_key,
                cache_payload,
                ttl_seconds=ttl_seconds,
            )
            result_state["cache_written"] = True
            logger.debug(
                "Stored grounded response in query cache",
                extra={
                    "knowledge_base_version": knowledge_base_version,
                    "ttl_seconds": ttl_seconds,
                },
            )
        except Exception as exc:
            result_state["cache_write_error"] = str(exc)
            logger.exception(
                "Query cache write failed",
                extra={
                    "knowledge_base_version": knowledge_base_version,
                },
            )

        return result_state

    return node
