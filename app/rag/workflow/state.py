from typing import Any, TypedDict

from langchain_core.documents import Document


class QueryState(TypedDict, total=False):
    """
    Shared state for the ContextOps LangGraph query workflow.

    Design principles:
    - Keep only bounded conversation context here.
    - Full conversation history remains in persistent storage.
    - Query execution is asynchronous.
    - Request-specific state lives here; providers remain shared.
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
    contextualized_query: str
    query_rewritten: bool
    query_rewrite_fallback: bool

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
    retrieved_documents: list[Document]

    # ==========================================
    # Reranking
    # ==========================================
    reranked_documents: list[Document]

    # ==========================================
    # Retrieval validation
    # ==========================================
    retrieval_confidence: str | None
    retrieval_score: float | None
    retrieval_sufficient: bool
    retrieval_reason: str | None

    # ==========================================
    # Recovery
    # ==========================================
    retry_count: int
    recovery_strategy: str | None

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