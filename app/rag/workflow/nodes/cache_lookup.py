from collections.abc import Awaitable, Callable

from app.core.logging import get_logger
from app.rag.workflow.state import QueryState
from app.services.cache_key import build_query_cache_key
from app.services.cache_provider import CacheProvider


logger = get_logger(__name__)


def create_cache_lookup_node(
    cache_provider: CacheProvider,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """Create the version-aware query cache lookup node."""

    async def node(state: QueryState) -> QueryState:
        tenant_id = state["tenant_id"]
        retrieval_query = state.get("contextualized_query") or state["query"]
        filters = state.get("filters")

        try:
            knowledge_base_version = (
                await cache_provider.get_knowledge_base_version(
                    tenant_id
                )
            )

            cache_key = build_query_cache_key(
                tenant_id=tenant_id,
                query=retrieval_query,
                filters=filters,
                knowledge_base_version=knowledge_base_version,
            )

            entry = await cache_provider.get(cache_key)

        except Exception as exc:
            logger.exception(
                "Query cache lookup failed; continuing with retrieval",
                extra={"tenant_id": tenant_id},
            )
            return {
                **state,
                "knowledge_base_version": None,
                "cache_key": None,
                "cache_hit": False,
                "cached_response": None,
                "cache_error": str(exc),
            }

        if entry is None:
            logger.debug(
                "Query cache miss",
                extra={
                    "tenant_id": tenant_id,
                    "knowledge_base_version": knowledge_base_version,
                },
            )
            return {
                **state,
                "knowledge_base_version": knowledge_base_version,
                "cache_key": cache_key,
                "cache_hit": False,
                "cached_response": None,
                "cache_error": None,
            }

        cached_version = entry.value.get("knowledge_base_version")
        if cached_version is not None and str(cached_version) != str(knowledge_base_version):
            logger.warning(
                "Ignoring cache entry with stale knowledge-base generation",
                extra={
                    "tenant_id": tenant_id,
                    "knowledge_base_version": knowledge_base_version,
                    "cached_knowledge_base_version": str(cached_version),
                },
            )
            return {
                **state,
                "knowledge_base_version": knowledge_base_version,
                "cache_key": cache_key,
                "cache_hit": False,
                "cached_response": None,
                "cache_error": None,
            }

        logger.info(
            "Query cache hit",
            extra={
                "tenant_id": tenant_id,
                "knowledge_base_version": knowledge_base_version,
            },
        )

        return {
            **state,
            "knowledge_base_version": knowledge_base_version,
            "cache_key": cache_key,
            "cache_hit": True,
            "cached_response": entry.value,
            "cache_error": None,
        }

    return node
