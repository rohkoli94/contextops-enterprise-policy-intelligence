from collections.abc import Awaitable, Callable
from typing import Any

from app.config.settings import settings
from app.rag.retrieval.document_mapper import (
    retrieved_chunks_to_documents,
)
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.workflow.state import QueryState


def create_hybrid_retrieval_node(
    hybrid_retriever: HybridRetriever,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph hybrid retrieval node.

    Uses the contextualized query while preserving the original
    user query separately in QueryState.
    """

    async def node(state: QueryState) -> QueryState:
        retrieval_query = (
            state.get("contextualized_query")
            or state["query"]
        )

        results = await hybrid_retriever.aretrieve(
            query=retrieval_query,
            tenant_id=state["tenant_id"],
            top_k=settings.retrieval_top_k,
            filters=state.get("filters"),
        )

        return {
            **state,
            "retrieved_documents": (
                retrieved_chunks_to_documents(results)
            ),
        }

    return node