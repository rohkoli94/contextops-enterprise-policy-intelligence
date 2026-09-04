from typing import Any

from app.api.v1.query.schemas.query_filter import (
    QueryFilter,
)
from app.rag.workflow.graph import (
    create_query_graph,
)
from app.rag.workflow.state import QueryState


class QueryService:
    """
    Thin application service for executing the ContextOps
    LangGraph query workflow.

    The service is intentionally NOT responsible for:
        - retrieval
        - reranking
        - context construction
        - query rewriting
        - caching
        - LLM generation
        - grounding validation

    Those responsibilities belong to the LangGraph workflow.
    """

    def __init__(
        self,
        query_graph,
    ) -> None:
        """
        Store the already-compiled application-scoped graph.

        The graph is created once during FastAPI startup and
        reused across requests.
        """

        self.query_graph = query_graph

    # ========================================================
    # QUERY
    # ========================================================

    async def ask(
        self,
        *,
        question: str,
        tenant_id: str,
        conversation_id: str | None = None,
        filters: QueryFilter | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete ContextOps query workflow.

        Request-specific state is created here.

        The compiled LangGraph itself is shared across requests.
        """

        # ----------------------------------------------------
        # REQUEST VALIDATION
        # ----------------------------------------------------

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        # ----------------------------------------------------
        # FILTER CONVERSION
        # ----------------------------------------------------
        #
        # API layer uses the strongly typed QueryFilter model.
        #
        # LangGraph state keeps the transport-neutral dictionary
        # representation used by retrieval components.
        #

        retrieval_filters = (
            filters.model_dump(
                exclude_none=True
            )
            if filters
            else None
        )

        # ----------------------------------------------------
        # INITIAL QUERY STATE
        # ----------------------------------------------------

        initial_state: QueryState = {
            "query": question.strip(),
            "tenant_id": tenant_id.strip(),
            "conversation_id": conversation_id,
            "filters": retrieval_filters,
            "retry_count": 0,
        }

        # ----------------------------------------------------
        # EXECUTE LANGGRAPH
        # ----------------------------------------------------

        result = await self.query_graph.ainvoke(
            initial_state
        )

        return result