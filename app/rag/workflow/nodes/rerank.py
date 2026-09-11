from collections.abc import Awaitable, Callable

from app.config.settings import settings
from app.rag.retrieval.reranker import Reranker
from app.rag.workflow.state import QueryState


def create_rerank_node(
    reranker: Reranker,
    candidate_limit: int | None = None,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph reranking node.

    The node explicitly limits the hybrid-retrieval results to the
    configured reranker candidate pool before invoking the reranker.

    This keeps the workflow boundary aligned with the production
    retrieval design:
        hybrid retrieval -> top-N candidate pool -> reranker
    """

    configured_candidate_limit = (
        candidate_limit
        if candidate_limit is not None
        else settings.rerank_candidate_limit
    )

    if configured_candidate_limit <= 0:
        raise ValueError(
            "candidate_limit must be greater than zero."
        )

    async def node(state: QueryState) -> QueryState:
        query = (
            state.get("contextualized_query")
            or state["query"]
        )

        candidates = state.get(
            "retrieved_documents",
            [],
        )

        if not candidates:
            return {
                **state,
                "reranked_documents": [],
            }

        candidate_pool = candidates[
            :configured_candidate_limit
        ]

        reranked_candidates = await reranker.rerank(
            query=query,
            candidates=candidate_pool,
        )

        return {
            **state,
            "reranked_documents": reranked_candidates,
        }

    return node