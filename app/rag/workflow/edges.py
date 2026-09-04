from typing import Literal

from app.rag.workflow.state import QueryState


def route_after_cache_lookup(
    state: QueryState,
) -> Literal[
    "cached_response",
    "hybrid_retrieval",
]:
    """
    Route the workflow after cache lookup.

    Cache is an optimization only.

    HIT:
        use cached response

    MISS:
        continue through retrieval
    """

    if state.get("cache_hit") is True:
        return "cached_response"

    return "hybrid_retrieval"


def route_after_retrieval_validation(
    state: QueryState,
) -> Literal[
    "contextops",
    "retrieval_recovery",
]:
    """
    Route based on retrieval sufficiency.

    Day 18:
        recovery is structural.

    Day 19:
        recovery becomes fully implemented.
    """

    if state.get("retrieval_sufficient") is True:
        return "contextops"

    return "retrieval_recovery"