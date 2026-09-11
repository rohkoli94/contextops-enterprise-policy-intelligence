from collections.abc import Awaitable, Callable

from app.config.settings import settings
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.models import RetrievedChunk
from app.rag.workflow.state import QueryState


def create_hybrid_retrieval_node(
    hybrid_retriever: HybridRetriever,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph hybrid retrieval node.

    Uses the contextualized query for retrieval while preserving
    the original user query separately in QueryState.

    RetrievedChunk objects are intentionally preserved in workflow
    state so retrieval score, metadata, tenant information, and
    reranker signals remain available to downstream stages.
    """

    async def node(state: QueryState) -> QueryState:
        retrieval_query = (
            state.get("contextualized_query")
            or state["query"]
        )

        results: list[RetrievedChunk] = (
            await hybrid_retriever.aretrieve(
                query=retrieval_query,
                tenant_id=state["tenant_id"],
                top_k=max(
                    settings.retrieval_top_k,
                    settings.rerank_candidate_limit,
                ),
                filters=state.get("filters"),
            )
        )

        return {
            **state,
            "retrieved_documents": results,
        }

    return node