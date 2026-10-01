import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.api.v1.query.schemas.query_filter import (
    QueryFilter,
)
from app.config.settings import settings
from app.domain.conversation import (
    ConversationMessage,
    ConversationRole,
)
from app.observability.timing import (
    record_total_query_time,
)
from app.providers.llm.base import (
    LLMProvider,
    LLMRequest,
)
from app.rag.workflow.state import QueryState
from app.services.conversation_memory import (
    ConversationMemory,
)


logger = logging.getLogger(
    "contextops.query"
)


class QueryService:
    """
    Application service responsible for executing
    ContextOps query workflows.

    Responsibilities:

        - validate request inputs
        - create/maintain conversations
        - persist user messages
        - execute LangGraph
        - persist assistant messages
        - update rolling conversation summaries
        - record query duration
        - emit structured query logs
    """

    def __init__(
        self,
        query_graph: Any,
        conversation_memory: ConversationMemory,
        llm_provider: LLMProvider,
        shutdown_resources: list[Any] | None = None,
    ) -> None:

        self.query_graph = query_graph

        self.conversation_memory = (
            conversation_memory
        )

        self.llm_provider = (
            llm_provider
        )

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
        Execute a complete ContextOps query.

        Conversation lifecycle:

            1. ensure conversation
            2. save user message
            3. execute LangGraph
            4. save assistant message
            5. update rolling summary
        """

        # =====================================================
        # STEP 1 Ã¢â‚¬â€ INPUT VALIDATION
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
        # STEP 2 Ã¢â‚¬â€ NORMALIZE
        # =====================================================

        normalized_question = (
            question.strip()
        )

        normalized_tenant_id = (
            tenant_id.strip()
        )

        # =====================================================
        # STEP 3 Ã¢â‚¬â€ CONVERSATION
        # =====================================================

        if conversation_id:
            normalized_conversation_id = (
                conversation_id.strip()
            )

        else:
            normalized_conversation_id = str(
                uuid4()
            )

        normalized_conversation_id = (
            await self.conversation_memory.ensure_conversation(
                conversation_id=(
                    normalized_conversation_id
                ),
                tenant_id=(
                    normalized_tenant_id
                ),
                title=(
                    normalized_question[:100]
                ),
            )
        )

        # =====================================================
        # STEP 4 Ã¢â‚¬â€ SAVE USER MESSAGE
        # =====================================================

        user_message = ConversationMessage(
            message_id=str(
                uuid4()
            ),
            conversation_id=(
                normalized_conversation_id
            ),
            role=ConversationRole.USER,
            content=normalized_question,
            created_at=datetime.now(
                timezone.utc
            ),
        )

        await self.conversation_memory.save_message(
            message=user_message,
            tenant_id=normalized_tenant_id,
        )

        # =====================================================
        # STEP 5 Ã¢â‚¬â€ NORMALIZE FILTERS
        # =====================================================

        retrieval_filters = (
            filters.model_dump(
                exclude_none=True,
            )
            if filters
            else None
        )

        # =====================================================
        # STEP 6 Ã¢â‚¬â€ TRACE ID
        # =====================================================

        trace_run_id = uuid4()

        langfuse_trace_id = None
        langfuse_handler = None

        if (
            settings.langfuse_tracing
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        ):
            langfuse = get_client()
            langfuse_trace_id = langfuse.create_trace_id(
                seed=str(trace_run_id)
            )
            langfuse_handler = CallbackHandler()

        # =====================================================
        # STEP 7 Ã¢â‚¬â€ INITIAL STATE
        # =====================================================

        initial_state: QueryState = {
            "query": normalized_question,
            "tenant_id": normalized_tenant_id,
            "conversation_id": (
                normalized_conversation_id
            ),
            "filters": retrieval_filters,
            "retry_count": 0,
            "timings": {
                "stages": {},
                "total_ms": None,
            },
        }

        # =====================================================
        # STEP 8 Ã¢â‚¬â€ LANGFUSE
        # =====================================================

        trace_tags = [
            "contextops",
            "rag",
            "langgraph",
            settings.environment,
        ]

        trace_metadata: dict[str, Any] = {
            "application": "contextops",
            "environment": settings.environment,
            "tenant_id": normalized_tenant_id,
            "conversation_id": (
                normalized_conversation_id
            ),
        }

        started_at = perf_counter()

        result: QueryState = {
            **initial_state
        }

        try:

            # =================================================
            # STEP 9 Ã¢â‚¬â€ LANGGRAPH
            # =================================================

            if langfuse_handler is not None:
                langfuse = get_client()
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="contextops-query",
                    trace_context={
                        "trace_id": langfuse_trace_id,
                    },
                    input={
                        "query": normalized_question,
                    },
                ) as root_span:
                    with propagate_attributes(
                        trace_name="contextops-query",
                        session_id=normalized_conversation_id,
                        tags=trace_tags,
                        metadata=trace_metadata,
                        environment=settings.environment,
                    ):
                        result = await self.query_graph.ainvoke(
                            initial_state,
                            config={
                                "callbacks": [langfuse_handler],
                                "tags": trace_tags,
                                "metadata": trace_metadata,
                            },
                        )

                    root_span.update(
                        output={
                            "answer": result.get("answer", ""),
                            "grounding_status": result.get(
                                "grounding_status"
                            ),
                        }
                    )

            else:
                result = await self.query_graph.ainvoke(
                    initial_state,
                    config={
                        "tags": trace_tags,
                        "metadata": trace_metadata,
                    },
                )

            # =================================================
            # STEP 10 Ã¢â‚¬â€ TOTAL TIME
            # =================================================

            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            result = record_total_query_time(
                result,
                duration_ms,
            )

            # =================================================
            # STEP 11 Ã¢â‚¬â€ TRACE ID
            # =================================================

            result = {
                **result,
                "conversation_id": (
                    normalized_conversation_id
                ),
                "langfuse_trace_id": langfuse_trace_id,
            }

            # =================================================
            # STEP 12 Ã¢â‚¬â€ SAVE ASSISTANT MESSAGE
            # =================================================

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            ).strip()

            if answer:

                assistant_message = (
                    ConversationMessage(
                        message_id=str(
                            uuid4()
                        ),
                        conversation_id=(
                            normalized_conversation_id
                        ),
                        role=(
                            ConversationRole.ASSISTANT
                        ),
                        content=answer,
                        created_at=datetime.now(
                            timezone.utc
                        ),
                    )
                )

                await (
                    self.conversation_memory
                    .save_message(
                        message=assistant_message,
                        tenant_id=(
                            normalized_tenant_id
                        ),
                    )
                )

                # =============================================
                # STEP 13 Ã¢â‚¬â€ UPDATE SUMMARY
                # =============================================

                await self._update_conversation_summary(
                    conversation_id=(
                        normalized_conversation_id
                    ),
                    tenant_id=(
                        normalized_tenant_id
                    ),
                )

            timings = result.get(
                "timings",
                {},
            )

            # =================================================
            # STEP 14 Ã¢â‚¬â€ LOGGING
            # =================================================

            logger.info(
                "ContextOps query completed",
                extra={
                    "tenant_id": (
                        normalized_tenant_id
                    ),
                    "conversation_id": (
                        normalized_conversation_id
                    ),
                    "langfuse_trace_id": langfuse_trace_id,
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

            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            logger.exception(
                "ContextOps query failed",
                extra={
                    "tenant_id": (
                        normalized_tenant_id
                    ),
                    "conversation_id": (
                        normalized_conversation_id
                    ),
                    "langfuse_trace_id": langfuse_trace_id,
                    "total_ms": round(
                        duration_ms,
                        3,
                    ),
                },
            )

            raise

    async def _update_conversation_summary(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
    ) -> None:
        """
        Generate and persist the rolling conversation summary.

        The summary is deliberately bounded around:

            existing summary
            +
            recent conversation messages

        so the summarization prompt does not grow
        indefinitely.
        """

        try:

            context = (
                await self.conversation_memory
                .get_context(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    recent_message_limit=(
                        settings.conversation_recent_message_limit
                    ),
                )
            )

            existing_summary = (
                context.summary
                or "No previous summary exists."
            )

            recent_lines = []

            for message in (
                context.recent_messages
            ):
                recent_lines.append(
                    f"{message.role.value}: "
                    f"{message.content}"
                )

            recent_context = (
                "\n".join(recent_lines)
                if recent_lines
                else "No recent messages."
            )

            system_prompt = """
You maintain a concise rolling summary for an enterprise
policy assistant conversation.

Summarize only the conversation context provided.

Preserve:
- the main policy/document topic
- important entities and subjects
- user intent
- important facts explicitly discussed
- unresolved questions or follow-ups

Do not invent information.

Do not answer the user's question.

Return only the updated summary.

Keep the summary concise and suitable for future
query contextualization.
""".strip()

            user_prompt = f"""
Existing conversation summary:
{existing_summary}

Recent conversation:
{recent_context}

Updated conversation summary:
""".strip()

            response = await (
                self.llm_provider.agenerate(
                    LLMRequest(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                )
            )

            summary = (
                response.content.strip()
            )

            if not summary:
                return

            await (
                self.conversation_memory
                .update_summary(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    summary=summary,
                    message_boundary=len(
                        context.recent_messages
                    ),
                )
            )

        except Exception:
            # Summary persistence must not make a successful
            # user query fail.
            logger.exception(
                "Failed to update conversation summary",
                extra={
                    "conversation_id": (
                        conversation_id
                    ),
                    "tenant_id": tenant_id,
                },
            )

    async def aclose(self) -> None:
        """
        Close long-lived asynchronous resources.
        """

        for resource in reversed(
            self.shutdown_resources
        ):
            close_method = getattr(
                resource,
                "aclose",
                None,
            )

            if close_method is not None:
                try:
                    await close_method()

                except Exception:
                    logger.exception(
                        "Failed to close application "
                        "resource: %r",
                        resource,
                    )
                continue

            close_method = getattr(
                resource,
                "close",
                None,
            )

            if close_method is not None:

                try:
                    result = close_method()

                    if hasattr(
                        result,
                        "__await__",
                    ):
                        await result

                except Exception:
                    logger.exception(
                        "Failed to close application "
                        "resource: %r",
                        resource,
                    )
