import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.api.v1.query.schemas.query_filter import QueryFilter
from app.config.settings import settings
from app.observability.timing import record_total_query_time
from app.rag.workflow.state import QueryState


logger = logging.getLogger("contextops.query")


class QueryService:
    """
    Application service responsible for executing ContextOps
    query workflows.

    Responsibilities:

        - validate request-level inputs
        - normalize query inputs
        - construct initial QueryState
        - create a trace/run ID
        - invoke the compiled LangGraph
        - pass LangSmith trace metadata
        - record total query duration
        - emit structured query logs
    """

    def __init__(
        self,
        query_graph: Any,
        shutdown_resources: list[Any] | None = None,
    ) -> None:
        """
        Initialize the query service.

        Args:
            query_graph:
                Compiled LangGraph used to execute queries.

            shutdown_resources:
                Long-lived resources that must be closed during
                application shutdown.
        """

        self.query_graph = query_graph

        self.shutdown_resources = (
            shutdown_resources
            if shutdown_resources is not None
            else []
        )

    async def ask(
        self,
        *,
        question: str,
        tenant_id: str,
        conversation_id: str | None = None,
        filters: QueryFilter | None = None,
    ) -> QueryState:
        """
        Execute a ContextOps query.

        A unique run ID is generated for every query and passed
        to LangGraph so the entire workflow can be associated
        with one LangSmith root trace.
        """

        # =====================================================
        # STEP 1 — INPUT VALIDATION
        # =====================================================

        if not isinstance(question, str):
            raise ValueError(
                "Question must be a string."
            )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not isinstance(tenant_id, str):
            raise ValueError(
                "Tenant ID must be a string."
            )

        if not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        # =====================================================
        # STEP 2 — NORMALIZE INPUTS
        # =====================================================

        normalized_question = (
            question.strip()
        )

        normalized_tenant_id = (
            tenant_id.strip()
        )

        # =====================================================
        # STEP 3 — NORMALIZE RETRIEVAL FILTERS
        # =====================================================

        retrieval_filters = (
            filters.model_dump(
                exclude_none=True,
            )
            if filters
            else None
        )

        # =====================================================
        # STEP 4 — CREATE TRACE RUN ID
        # =====================================================

        trace_run_id = uuid4()

        # =====================================================
        # STEP 5 — INITIAL WORKFLOW STATE
        # =====================================================

        initial_state: QueryState = {
            "query": normalized_question,
            "tenant_id": normalized_tenant_id,
            "conversation_id": conversation_id,
            "filters": retrieval_filters,
            "retry_count": 0,
            "timings": {
                "stages": {},
                "total_ms": None,
            },
        }

        # =====================================================
        # STEP 6 — LANGSMITH TRACE TAGS
        # =====================================================

        trace_tags = [
            "contextops",
            "rag",
            "langgraph",
            settings.environment,
        ]

        # =====================================================
        # STEP 7 — LANGSMITH TRACE METADATA
        # =====================================================

        trace_metadata: dict[str, Any] = {
            "application": "contextops",
            "environment": settings.environment,
            "tenant_id": normalized_tenant_id,
            "conversation_id": conversation_id,
        }

        started_at = perf_counter()

        result: QueryState = {
            **initial_state
        }

        try:
            # =================================================
            # STEP 8 — INVOKE LANGGRAPH
            # =================================================
            #
            # Explicit run_id makes this invocation the root
            # trace associated with this query.
            #

            result = await self.query_graph.ainvoke(
                initial_state,
                config={
                    "run_id": trace_run_id,
                    "tags": trace_tags,
                    "metadata": trace_metadata,
                },
            )

            # =================================================
            # STEP 9 — RECORD TOTAL QUERY TIME
            # =================================================

            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            result = record_total_query_time(
                result,
                duration_ms,
            )

            # =================================================
            # STEP 10 — EXPOSE TRACE RUN ID
            # =================================================
            #
            # Keep the trace identifier available to evaluation
            # and observability consumers without putting the
            # raw query into custom metadata.
            #

            result = {
                **result,
                "langsmith_run_id": str(
                    trace_run_id
                ),
            }

            timings = result.get(
                "timings",
                {},
            )

            # =================================================
            # STEP 11 — STRUCTURED QUERY LOGGING
            # =================================================

            logger.info(
                "ContextOps query completed",
                extra={
                    "tenant_id": normalized_tenant_id,
                    "conversation_id": conversation_id,
                    "langsmith_run_id": str(
                        trace_run_id
                    ),
                    "cache_hit": result.get(
                        "cache_hit",
                        False,
                    ),
                    "cache_written": result.get(
                        "cache_written",
                        False,
                    ),
                    "retrieval_confidence": result.get(
                        "retrieval_confidence",
                    ),
                    "retrieval_sufficient": result.get(
                        "retrieval_sufficient",
                    ),
                    "grounding_status": result.get(
                        "grounding_status",
                    ),
                    "query_rewritten": result.get(
                        "query_rewritten",
                        False,
                    ),
                    "context_token_count": result.get(
                        "context_token_count",
                        0,
                    ),
                    "context_pii_detected": result.get(
                        "context_pii_detected",
                        False,
                    ),
                    "context_compressed": result.get(
                        "context_compressed",
                        False,
                    ),
                    "timings": timings,
                },
            )

            return result

        except Exception:
            # =================================================
            # ERROR TIMING
            # =================================================

            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            # =================================================
            # STRUCTURED ERROR LOGGING
            # =================================================

            logger.exception(
                "ContextOps query failed",
                extra={
                    "tenant_id": normalized_tenant_id,
                    "conversation_id": conversation_id,
                    "langsmith_run_id": str(
                        trace_run_id
                    ),
                    "total_ms": round(
                        duration_ms,
                        3,
                    ),
                },
            )

            raise

    async def aclose(self) -> None:
        """
        Close all long-lived asynchronous resources owned by
        the query service.

        Resources are closed in reverse creation order so that
        dependencies are released safely.
        """

        for resource in reversed(
            self.shutdown_resources
        ):
            close_method = getattr(
                resource,
                "aclose",
                None,
            )

            if close_method is None:
                close_method = getattr(
                    resource,
                    "close",
                    None,
                )

            if close_method is None:
                continue

            result = close_method()

            if hasattr(
                result,
                "__await__",
            ):
                await result