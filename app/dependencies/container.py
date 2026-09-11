from app.db.session import AsyncSessionLocal

from app.dependencies.rag import (
    get_hybrid_retriever,
)
from app.guardrails import (
    AuthorizationGuard,
    PromptInjectionGuard,
    RegexPIIAnalyzer,
    TenantIsolationGuard,
)
from app.providers.llm.microsoft_foundry import (
    MicrosoftFoundryProvider,
)
from app.rag.workflow.graph import (
    create_query_graph,
)
from app.repositories.conversation import (
    ConversationRepository,
)
from app.rag.retrieval.baseline_grounding_validator import (
    BaselineGroundingValidator,
)
from app.rag.retrieval.baseline_retrieval_validator import (
    BaselineRetrievalValidator,
)
from app.rag.retrieval.fastembed_reranker import (
    FastEmbedReranker,
)
from app.services.conversation_memory import (
    PostgresConversationMemory,
)
from app.services.microsoft_foundry_query_rewriter import (
    MicrosoftFoundryQueryRewriter,
)
from app.services.no_op_cache import (
    NoOpCacheProvider,
)
from app.services.query_service import (
    QueryService,
)


def create_query_service() -> QueryService:
    """
    Compose the shared query application graph.

    This function is called once during FastAPI startup.

    Long-lived/shared components:

        MicrosoftFoundryProvider
                +
        HybridRetriever
                +
        ConversationMemory
                +
        QueryRewriter
                +
        CacheProvider
                +
        Guardrails
                +
        Reranker
                +
        Validators
                ↓
           LangGraph
                ↓
          QueryService

    Request-specific state is created inside QueryService.ask().
    """

    # ========================================================
    # SHARED LLM PROVIDER
    # ========================================================

    llm_provider = MicrosoftFoundryProvider()

    # ========================================================
    # SHARED HYBRID RETRIEVER
    # ========================================================

    hybrid_retriever = get_hybrid_retriever()

    # ========================================================
    # SHARED CONVERSATION REPOSITORY
    # ========================================================
    #
    # IMPORTANT:
    # AsyncSessionLocal is a session factory.
    #
    # We do NOT create one AsyncSession here and keep it
    # inside the application-scoped service.
    #

    conversation_repository = ConversationRepository(
        session_factory=AsyncSessionLocal,
    )

    # ========================================================
    # SHARED CONVERSATION MEMORY
    # ========================================================

    conversation_memory = PostgresConversationMemory(
        repository=conversation_repository,
    )

    # ========================================================
    # SHARED QUERY REWRITER
    # ========================================================

    query_rewriter = MicrosoftFoundryQueryRewriter(
        llm_provider=llm_provider,
    )

    # ========================================================
    # SHARED CACHE
    # ========================================================
    #
    # Day 18:
    #     NoOpCacheProvider
    #
    # Later:
    #     RedisCacheProvider
    #

    cache_provider = NoOpCacheProvider()

    # ========================================================
    # SHARED GUARDRAILS
    # ========================================================

    authorization_guard = AuthorizationGuard()

    tenant_isolation_guard = (
        TenantIsolationGuard()
    )

    prompt_injection_guard = (
        PromptInjectionGuard()
    )

    pii_analyzer = RegexPIIAnalyzer()

    # ========================================================
    # SHARED BASELINE RAG COMPONENTS
    # ========================================================
    

    reranker = FastEmbedReranker()

    retrieval_validator = (
        BaselineRetrievalValidator()
    )

    grounding_validator = (
        BaselineGroundingValidator()
    )

    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    query_graph = create_query_graph(
        llm_provider=llm_provider,
        hybrid_retriever=hybrid_retriever,
        conversation_memory=conversation_memory,
        query_rewriter=query_rewriter,
        cache_provider=cache_provider,
        authorization_guard=authorization_guard,
        tenant_isolation_guard=tenant_isolation_guard,
        prompt_injection_guard=prompt_injection_guard,
        pii_analyzer=pii_analyzer,
        reranker=reranker,
        retrieval_validator=retrieval_validator,
        grounding_validator=grounding_validator,
    )

    # ========================================================
    # QUERY SERVICE
    # ========================================================

    return QueryService(
        query_graph=query_graph,
    )