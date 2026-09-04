from collections.abc import Awaitable, Callable

from app.rag.retrieval.retrieval_validator import (
    RetrievalValidator,
)
from app.rag.workflow.state import QueryState


def create_retrieval_validation_node(
    retrieval_validator: RetrievalValidator,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph retrieval-validation node.

    The node evaluates reranked documents and stores the
    evaluation result in QueryState.
    """

    async def node(state: QueryState) -> QueryState:
        query = (
            state.get("contextualized_query")
            or state["query"]
        )

        documents = state.get(
            "reranked_documents",
            [],
        )

        evaluation = await retrieval_validator.evaluate(
            query=query,
            documents=documents,
        )

        return {
            **state,
            "retrieval_sufficient": evaluation.sufficient,
            "retrieval_confidence": evaluation.confidence,
            "retrieval_score": evaluation.score,
            "retrieval_reason": evaluation.reason,
        }

    return node