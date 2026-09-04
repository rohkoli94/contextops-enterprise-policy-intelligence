from collections.abc import Awaitable, Callable

from app.rag.retrieval.grounding_validator import (
    GroundingValidator,
)
from app.rag.workflow.state import QueryState


def create_grounding_validation_node(
    grounding_validator: GroundingValidator,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph grounding-validation node.

    The generated answer is checked against the same evidence
    used to construct the LLM context.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        query = state["query"]
        answer = state.get(
            "answer",
            "",
        )

        documents = state.get(
            "reranked_documents",
            [],
        )

        evaluation = await grounding_validator.evaluate(
            query=query,
            answer=answer,
            documents=documents,
        )

        return {
            **state,
            "grounding_status": (
                "grounded"
                if evaluation.grounded
                else "not_grounded"
            ),
            "grounding_reason": evaluation.reason,
            "grounding_supported_sources": (
                evaluation.supported_sources
            ),
        }

    return node