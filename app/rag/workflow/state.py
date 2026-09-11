from typing import Any, TypedDict

from app.rag.retrieval.models import RetrievedChunk


class QueryState(TypedDict, total=False):
    """
    Shared state for the ContextOps LangGraph query workflow.

    Design principles:
    - Keep only bounded conversation context here.
    - Full conversation history remains in persistent storage.
    - Query execution is asynchronous.
    - Request-specific state lives here; providers remain shared.
    - The original user query is never overwritten.
    - Recovery state is explicitly tracked and bounded.
    """

    # ==========================================
    # Request / identity
    # ==========================================
    query: str
    tenant_id: str
    conversation_id: str | None
    filters: dict[str, Any] | None

    # ==========================================
    # Input security / guardrails
    # ==========================================
    input_valid: bool
    authorization_valid: bool
    tenant_valid: bool
    prompt_injection_detected: bool
    input_pii_detected: bool
    input_guardrail_action: str | None
    guardrail_reason: str | None

    # ==========================================
    # Conversation context
    # ==========================================
    recent_messages: list[dict[str, Any]]
    conversation_summary: str | None

    # ==========================================
    # Query intelligence
    # ==========================================
    #
    # query:
    #     Original user question.
    #
    # contextualized_query:
    #     Conversation-aware retrieval query.
    #
    # recovery_query:
    #     Recovery-only reformulated query.
    #
    # The original query must never be replaced by either.
    # ==========================================
    contextualized_query: str
    query_rewritten: bool
    query_rewrite_fallback: bool
    recovery_query: str | None

    # ==========================================
    # Cache
    # ==========================================
    cache_key: str | None
    cache_hit: bool
    cached_response: dict[str, Any] | None
    cache_error: str | None

    # ==========================================
    # Retrieval
    # ==========================================
    retrieved_documents: list[RetrievedChunk]

    # ==========================================
    # Reranking
    # ==========================================
    reranked_documents: list[RetrievedChunk]

    # ==========================================
    # Retrieval validation
    # ==========================================
    retrieval_confidence: str | None
    retrieval_score: float | None
    retrieval_sufficient: bool
    retrieval_reason: str | None
    retrieval_signals: dict[str, Any]

    # ==========================================
    # Recovery
    # ==========================================
    #
    # retry_count:
    #     Number of recovery attempts already performed.
    #
    # recovery_strategy:
    #     Strategy used for recovery or safe abstention.
    #
    # recovery_success:
    #     True only when recovery produced sufficient
    #     evidence that can continue to ContextOps.
    #
    # recovery_error:
    #     Optional operational error captured during recovery.
    # ==========================================
    retry_count: int
    recovery_strategy: str | None
    recovery_success: bool
    recovery_error: str | None

    # ==========================================
    # ContextOps
    # ==========================================
    context: str
    citations: list[dict[str, Any]]

    # ==========================================
    # LLM
    # ==========================================
    answer: str

    # ==========================================
    # Output validation
    # ==========================================
    output_pii_detected: bool
    grounding_status: str | None
    grounding_reason: str | None
    grounding_supported_sources: list[int]
    final_response_metadata: dict[str, Any]