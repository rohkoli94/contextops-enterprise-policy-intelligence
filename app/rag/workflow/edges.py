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

    Sufficient evidence:
        continue to ContextOps.

    Insufficient evidence:
        attempt bounded retrieval recovery.
    """

    if state.get("retrieval_sufficient") is True:
        return "contextops"

    return "retrieval_recovery"


def route_after_retrieval_recovery(
    state: QueryState,
) -> Literal[
    "contextops",
    "response",
]:
    """
    Route after a bounded retrieval-recovery attempt.

    Recovery success:
        recovered evidence is available, so continue to
        ContextOps and then the LLM.

    Recovery failure:
        no sufficient evidence is available, so continue
        directly to the response node with safe abstention.

    This prevents weak evidence from reaching the LLM.
    """

    if state.get("recovery_success") is True:
        return "contextops"

    return "response"