from collections.abc import Awaitable, Callable

from app.rag.workflow.state import QueryState


def create_input_validation_node() -> (
    Callable[[QueryState], Awaitable[QueryState]]
):
    """
    Create the deterministic input-validation node.

    Validation occurs before security, conversation, cache,
    retrieval, or LLM processing.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        query = state.get(
            "query",
            "",
        )

        tenant_id = state.get(
            "tenant_id",
            "",
        )

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        filters = state.get("filters")

        if filters is not None and not isinstance(
            filters,
            dict,
        ):
            raise ValueError(
                "filters must be a dictionary."
            )

        return {
            **state,
            "query": query.strip(),
            "tenant_id": tenant_id.strip(),
            "input_valid": True,
        }

    return node