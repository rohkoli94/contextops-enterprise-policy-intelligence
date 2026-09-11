from collections.abc import Awaitable, Callable

from app.config.settings import settings
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.reranker import Reranker
from app.rag.retrieval.retrieval_validator import (
    RetrievalValidator,
)
from app.rag.workflow.state import QueryState
from app.services.query_rewriter import QueryRewriter


DEFAULT_MAX_RECOVERY_RETRIES = 1


def create_retrieval_recovery_node(
    *,
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    retrieval_validator: RetrievalValidator,
    query_rewriter: QueryRewriter,
    max_retries: int = DEFAULT_MAX_RECOVERY_RETRIES,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the Day 19 retrieval-recovery node.

    Recovery performs at most ``max_retries`` attempts.

    Recovery flow:

        failed retrieval validation
                    ↓
             query reformulation
                    ↓
              broader retrieval
                    ↓
                  rerank
                    ↓
             re-validation
                    ↓
          sufficient / insufficient

    If the retry limit has already been reached, the node
    performs safe abstention instead of sending weak evidence
    to the LLM.

    Tenant ID and retrieval filters are always preserved.
    """

    if max_retries < 0:
        raise ValueError(
            "max_retries cannot be negative."
        )

    async def node(
        state: QueryState,
    ) -> QueryState:
        retry_count = int(
            state.get(
                "retry_count",
                0,
            )
        )

        # ----------------------------------------------------
        # Retry limit
        # ----------------------------------------------------

        if retry_count >= max_retries:
            return _safe_abstention(
                state=state,
                retry_count=retry_count,
            )

        # ----------------------------------------------------
        # Original query must never be lost
        # ----------------------------------------------------

        original_query = state["query"].strip()

        retrieval_query = (
            state.get("contextualized_query")
            or original_query
        ).strip()

        if not retrieval_query:
            return _safe_abstention(
                state=state,
                retry_count=retry_count,
            )

        # ----------------------------------------------------
        # Query reformulation
        # ----------------------------------------------------
        #
        # We explicitly request a broader retrieval query while
        # preserving the conversation context already available
        # in QueryState.
        #
        # The original user query remains untouched in:
        #     state["query"]
        #
        # The reformulated query is retrieval-only.
        # ----------------------------------------------------

        recovery_prompt = (
            "Create a broader enterprise policy retrieval query "
            "for the user's question. Preserve the important "
            "policy terms, entities, dates, roles, and constraints. "
            "Use related terminology and expand the wording so "
            "relevant policy evidence is easier to retrieve. "
            "Return only the retrieval query.\n\n"
            f"Original user question:\n{original_query}\n\n"
            f"Previous retrieval query:\n{retrieval_query}"
        )

        try:
            reformulated_query = (
                await query_rewriter.rewrite(
                    query=recovery_prompt,
                    conversation_summary=state.get(
                        "conversation_summary"
                    ),
                    recent_messages=state.get(
                        "recent_messages",
                        [],
                    ),
                )
            )

            reformulated_query = (
                reformulated_query.strip()
            )

        except Exception:
            reformulated_query = ""

        # ----------------------------------------------------
        # Deterministic fallback
        # ----------------------------------------------------

        if not reformulated_query:
            reformulated_query = (
                f"{retrieval_query} "
                f"{original_query}"
            ).strip()

        # Prevent accidental empty / duplicate recovery query.
        if not reformulated_query:
            return _safe_abstention(
                state=state,
                retry_count=retry_count,
            )

        # ----------------------------------------------------
        # Broader retrieval
        # ----------------------------------------------------
        #
        # We retrieve more candidates during recovery so the
        # reranker has a wider evidence pool.
        #
        # Security invariants:
        #   tenant_id unchanged
        #   filters unchanged
        # ----------------------------------------------------

        recovery_top_k = max(
            settings.retrieval_top_k * 2,
            settings.rerank_candidate_limit,
        )

        try:
            retrieved_documents: list[RetrievedChunk] = (
                await hybrid_retriever.aretrieve(
                    query=reformulated_query,
                    tenant_id=state["tenant_id"],
                    top_k=recovery_top_k,
                    filters=state.get("filters"),
                )
            )
        except Exception as exc:
            return {
                **state,
                "retry_count": retry_count + 1,
                "recovery_strategy": "retrieval_error_safe_abstention",
                "recovery_query": reformulated_query,
                "recovery_error": str(exc),
                "answer": (
                    "I could not find sufficient policy evidence "
                    "to answer this question reliably."
                ),
                "citations": [],
                "grounding_status": "not_grounded",
                "grounding_reason": (
                    "Recovery retrieval was unavailable."
                ),
                "grounding_supported_sources": [],
            }

        # ----------------------------------------------------
        # No recovery evidence
        # ----------------------------------------------------

        if not retrieved_documents:
            return _safe_abstention(
                state=state,
                retry_count=retry_count + 1,
                recovery_query=reformulated_query,
                reason=(
                    "Recovery retrieval returned no documents."
                ),
            )

        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        candidate_pool = retrieved_documents[
            :settings.rerank_candidate_limit
        ]

        # ----------------------------------------------------
        # Re-ranking
        # ----------------------------------------------------

        try:
            reranked_documents = await reranker.rerank(
                query=reformulated_query,
                candidates=candidate_pool,
            )
        except Exception as exc:
            return _safe_abstention(
                state=state,
                retry_count=retry_count + 1,
                recovery_query=reformulated_query,
                reason=(
                    "Recovery reranking was unavailable: "
                    f"{exc}"
                ),
            )

        # ----------------------------------------------------
        # Re-validation
        # ----------------------------------------------------

        try:
            evaluation = (
                await retrieval_validator.evaluate(
                    query=reformulated_query,
                    documents=reranked_documents,
                )
            )
        except Exception as exc:
            return _safe_abstention(
                state=state,
                retry_count=retry_count + 1,
                recovery_query=reformulated_query,
                reason=(
                    "Recovery validation was unavailable: "
                    f"{exc}"
                ),
            )

        new_retry_count = retry_count + 1

        # ----------------------------------------------------
        # Recovery succeeded
        # ----------------------------------------------------

        if evaluation.sufficient:
            return {
                **state,
                "retry_count": new_retry_count,
                "recovery_strategy": "query_reformulation",
                "recovery_query": reformulated_query,
                "retrieved_documents": retrieved_documents,
                "reranked_documents": reranked_documents,
                "retrieval_sufficient": evaluation.sufficient,
                "retrieval_confidence": evaluation.confidence,
                "retrieval_score": evaluation.score,
                "retrieval_reason": evaluation.reason,
                "retrieval_signals": evaluation.signals,
                "recovery_success": True,
            }

        # ----------------------------------------------------
        # Recovery failed
        # ----------------------------------------------------

        return _safe_abstention(
            state={
                **state,
                "retry_count": new_retry_count,
                "recovery_query": reformulated_query,
                "retrieved_documents": retrieved_documents,
                "reranked_documents": reranked_documents,
                "retrieval_sufficient": evaluation.sufficient,
                "retrieval_confidence": evaluation.confidence,
                "retrieval_score": evaluation.score,
                "retrieval_reason": evaluation.reason,
                "retrieval_signals": evaluation.signals,
            },
            retry_count=new_retry_count,
            recovery_query=reformulated_query,
            reason=(
                "Recovery retrieval did not provide sufficient "
                "evidence."
            ),
        )

    return node


def _safe_abstention(
    *,
    state: QueryState,
    retry_count: int,
    recovery_query: str | None = None,
    reason: str | None = None,
) -> QueryState:
    """
    Produce a safe final response without invoking the LLM.
    """

    return {
        **state,
        "retry_count": retry_count,
        "recovery_strategy": "safe_abstention",
        "recovery_query": recovery_query,
        "recovery_success": False,
        "answer": (
            "I could not find sufficient policy evidence "
            "to answer this question reliably."
        ),
        "citations": [],
        "grounding_status": "not_grounded",
        "grounding_reason": (
            reason
            or "Retrieval did not provide sufficient evidence."
        ),
        "grounding_supported_sources": [],
    }