from app.guardrails import (
    AuthorizationGuard,
    PIIAnalyzer,
    PromptInjectionGuard,
    TenantIsolationGuard,
)
from app.providers.llm.base import LLMProvider
from app.rag.retrieval.baseline_grounding_validator import (
    BaselineGroundingValidator,
)
from app.rag.retrieval.baseline_retrieval_validator import (
    BaselineRetrievalValidator,
)
from app.rag.retrieval.grounding_validator import (
    GroundingValidator,
)
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.pass_through_reranker import (
    PassThroughReranker,
)
from app.rag.retrieval.reranker import Reranker
from app.rag.retrieval.retrieval_validator import (
    RetrievalValidator,
)
from app.rag.workflow.edges import (
    route_after_cache_lookup,
    route_after_retrieval_recovery,
    route_after_retrieval_validation,
)
from app.rag.workflow.nodes.cache_lookup import (
    create_cache_lookup_node,
)
from app.rag.workflow.nodes.cached_response import (
    create_cached_response_node,
)
from app.rag.workflow.nodes.contextops import (
    create_contextops_node,
)
from app.rag.workflow.nodes.conversation_context import (
    create_conversation_context_node,
)
from app.rag.workflow.nodes.grounding import (
    create_grounding_validation_node,
)
from app.rag.workflow.nodes.input_validation import (
    create_input_validation_node,
)
from app.rag.workflow.nodes.llm_generation import (
    create_llm_generation_node,
)
from app.rag.workflow.nodes.query_contextualization import (
    create_query_contextualization_node,
)
from app.rag.workflow.nodes.retrieval import (
    create_hybrid_retrieval_node,
)
from app.rag.workflow.nodes.retrieval_recovery import (
    create_retrieval_recovery_node,
)
from app.rag.workflow.nodes.retrieval_validation import (
    create_retrieval_validation_node,
)
from app.rag.workflow.nodes.rerank import (
    create_rerank_node,
)
from app.rag.workflow.nodes.response import (
    create_response_node,
)
from app.rag.workflow.nodes.security import (
    create_security_guardrails_node,
)
from app.rag.workflow.state import QueryState
from app.services.cache_provider import CacheProvider
from app.services.conversation_memory import ConversationMemory
from app.services.query_rewriter import QueryRewriter
from langgraph.graph import END, START, StateGraph


def create_query_graph(
    *,
    llm_provider: LLMProvider,
    hybrid_retriever: HybridRetriever,
    conversation_memory: ConversationMemory,
    query_rewriter: QueryRewriter,
    cache_provider: CacheProvider,
    authorization_guard: AuthorizationGuard,
    tenant_isolation_guard: TenantIsolationGuard,
    prompt_injection_guard: PromptInjectionGuard,
    pii_analyzer: PIIAnalyzer,
    reranker: Reranker | None = None,
    retrieval_validator: RetrievalValidator | None = None,
    grounding_validator: GroundingValidator | None = None,
):
    """
    Build and compile the ContextOps query workflow.

    Day 18:
        Complete end-to-end orchestration.

    Day 19:
        Real reranking, candidate-pool limiting, retrieval
        confidence evaluation, bounded recovery, query
        reformulation, re-retrieval, re-ranking, and
        safe abstention.

    Day 20:
        Production hardening, observability, caching, and
        evaluation improvements.
    """

    # --------------------------------------------------------
    # Baseline components
    # --------------------------------------------------------

    reranker = (
        reranker
        or PassThroughReranker()
    )

    retrieval_validator = (
        retrieval_validator
        or BaselineRetrievalValidator()
    )

    grounding_validator = (
        grounding_validator
        or BaselineGroundingValidator()
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    input_validation_node = (
        create_input_validation_node()
    )

    security_node = (
        create_security_guardrails_node(
            authorization_guard=authorization_guard,
            tenant_isolation_guard=tenant_isolation_guard,
            prompt_injection_guard=prompt_injection_guard,
            pii_analyzer=pii_analyzer,
        )
    )

    conversation_context_node = (
        create_conversation_context_node(
            conversation_memory
        )
    )

    query_contextualization_node = (
        create_query_contextualization_node(
            query_rewriter
        )
    )

    cache_lookup_node = (
        create_cache_lookup_node(
            cache_provider
        )
    )

    cached_response_node = (
        create_cached_response_node()
    )

    hybrid_retrieval_node = (
        create_hybrid_retrieval_node(
            hybrid_retriever
        )
    )

    rerank_node = (
        create_rerank_node(
            reranker
        )
    )

    retrieval_validation_node = (
        create_retrieval_validation_node(
            retrieval_validator
        )
    )

    retrieval_recovery_node = (
        create_retrieval_recovery_node(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            retrieval_validator=retrieval_validator,
            query_rewriter=query_rewriter,
            max_retries=1,
        )
    )

    contextops_node = (
        create_contextops_node()
    )

    llm_generation_node = (
        create_llm_generation_node(
            llm_provider
        )
    )

    grounding_node = (
        create_grounding_validation_node(
            grounding_validator
        )
    )

    response_node = (
        create_response_node()
    )

    # --------------------------------------------------------
    # StateGraph
    # --------------------------------------------------------

    builder = StateGraph(QueryState)

    # --------------------------------------------------------
    # Register nodes
    # --------------------------------------------------------

    builder.add_node(
        "input_validation",
        input_validation_node,
    )

    builder.add_node(
        "security",
        security_node,
    )

    builder.add_node(
        "conversation_context",
        conversation_context_node,
    )

    builder.add_node(
        "query_contextualization",
        query_contextualization_node,
    )

    builder.add_node(
        "cache_lookup",
        cache_lookup_node,
    )

    builder.add_node(
        "cached_response",
        cached_response_node,
    )

    builder.add_node(
        "hybrid_retrieval",
        hybrid_retrieval_node,
    )

    builder.add_node(
        "rerank",
        rerank_node,
    )

    builder.add_node(
        "retrieval_validation",
        retrieval_validation_node,
    )

    builder.add_node(
        "retrieval_recovery",
        retrieval_recovery_node,
    )

    builder.add_node(
        "contextops",
        contextops_node,
    )

    builder.add_node(
        "llm_generation",
        llm_generation_node,
    )

    builder.add_node(
        "grounding",
        grounding_node,
    )

    builder.add_node(
        "response",
        response_node,
    )

    # --------------------------------------------------------
    # START → INPUT VALIDATION
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "input_validation",
    )

    # --------------------------------------------------------
    # NORMAL FLOW
    # --------------------------------------------------------

    builder.add_edge(
        "input_validation",
        "security",
    )

    builder.add_edge(
        "security",
        "conversation_context",
    )

    builder.add_edge(
        "conversation_context",
        "query_contextualization",
    )

    builder.add_edge(
        "query_contextualization",
        "cache_lookup",
    )

    # --------------------------------------------------------
    # CACHE ROUTING
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {
            "cached_response": "cached_response",
            "hybrid_retrieval": "hybrid_retrieval",
        },
    )

    builder.add_edge(
        "cached_response",
        "response",
    )

    # --------------------------------------------------------
    # INITIAL RETRIEVAL
    # --------------------------------------------------------

    builder.add_edge(
        "hybrid_retrieval",
        "rerank",
    )

    builder.add_edge(
        "rerank",
        "retrieval_validation",
    )

    # --------------------------------------------------------
    # INITIAL RETRIEVAL VALIDATION
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "retrieval_validation",
        route_after_retrieval_validation,
        {
            "contextops": "contextops",
            "retrieval_recovery": (
                "retrieval_recovery"
            ),
        },
    )

    # --------------------------------------------------------
    # RETRIEVAL RECOVERY
    # --------------------------------------------------------
    #
    # Recovery performs:
    #
    #   query reformulation
    #          ↓
    #   broader retrieval
    #          ↓
    #       top-20 pool
    #          ↓
    #        rerank
    #          ↓
    #      validation
    #
    # Successful recovery continues to ContextOps.
    #
    # Failed recovery goes directly to Response with
    # safe abstention, preventing unsupported evidence
    # from reaching the LLM.
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "retrieval_recovery",
        route_after_retrieval_recovery,
        {
            "contextops": "contextops",
            "response": "response",
        },
    )

    # --------------------------------------------------------
    # CONTEXT → LLM
    # --------------------------------------------------------

    builder.add_edge(
        "contextops",
        "llm_generation",
    )

    builder.add_edge(
        "llm_generation",
        "grounding",
    )

    # --------------------------------------------------------
    # GROUNDING → RESPONSE
    # --------------------------------------------------------

    builder.add_edge(
        "grounding",
        "response",
    )

    # --------------------------------------------------------
    # RESPONSE → END
    # --------------------------------------------------------

    builder.add_edge(
        "response",
        END,
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    return builder.compile()