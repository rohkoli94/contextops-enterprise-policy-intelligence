from app.rag.workflow.state import QueryState


async def retrieval_recovery_node(
    state: QueryState,
) -> QueryState:
    """
    Day 18 structural recovery.

    Until Day 19 implements real re-retrieval, weak retrieval
    results are handled with safe abstention.

    This prevents unsupported evidence from reaching the LLM.
    """

    return {
        **state,
        "recovery_strategy": "safe_abstention_pending_day19",
        "retry_count": state.get(
            "retry_count",
            0,
        ),
        "answer": (
            "I could not find sufficient policy evidence "
            "to answer this question reliably."
        ),
        "citations": [],
        "grounding_status": "not_grounded",
        "grounding_reason": (
            "Retrieval did not provide sufficient evidence."
        ),
        "grounding_supported_sources": [],
    }