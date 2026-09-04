from collections.abc import Awaitable, Callable

from app.rag.retrieval.reranker import Reranker
from app.rag.workflow.state import QueryState


def create_rerank_node(
    reranker: Reranker,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph reranking node.

    The node operates on the documents returned by hybrid
    retrieval and stores the reordered documents separately.
    """

    async def node(state: QueryState) -> QueryState:
        query = (
            state.get("contextualized_query")
            or state["query"]
        )

        documents = state.get(
            "retrieved_documents",
            [],
        )

        if not documents:
            return {
                **state,
                "reranked_documents": [],
            }

        reranked_documents = await reranker.rerank(
            query=query,
            documents=documents,
        )

        return {
            **state,
            "reranked_documents": reranked_documents,
        }

    return node