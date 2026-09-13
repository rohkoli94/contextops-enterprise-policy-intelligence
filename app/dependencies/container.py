from app.db.session import AsyncSessionLocal

from app.config.settings import settings
from app.dependencies.rag import (
    get_embedding_provider,
    get_hybrid_retriever,
    get_vector_store,
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
from app.rag.retrieval.baseline_grounding_validator import (
    BaselineGroundingValidator,
)
from app.rag.retrieval.baseline_retrieval_validator import (
    BaselineRetrievalValidator,
)
from app.rag.retrieval.fastembed_reranker import (
    FastEmbedReranker,
)
from app.rag.workflow.graph import (
    create_query_graph,
)
from app.repositories.conversation import (
    ConversationRepository,
)
from app.services.conversation_memory import (
    PostgresConversationMemory,
)
from app.services.microsoft_foundry_query_rewriter import (
    MicrosoftFoundryQueryRewriter,
)
from app.services.query_service import (
    QueryService,
)
from app.services.redis_cache import (
    RedisCacheProvider,
)


def create_query_service() -> QueryService:
    # ========================================================
    # PROVIDERS
    # ========================================================

    llm_provider = MicrosoftFoundryProvider()

    embedding_provider = get_embedding_provider()

    hybrid_retriever = get_hybrid_retriever()

    vector_store = get_vector_store()

    # ========================================================
    # CONVERSATION MEMORY
    # ========================================================

    conversation_repository = ConversationRepository(
        session_factory=AsyncSessionLocal,
    )

    conversation_memory = PostgresConversationMemory(
        repository=conversation_repository,
    )

    # ========================================================
    # QUERY REWRITER
    # ========================================================

    query_rewriter = MicrosoftFoundryQueryRewriter(
        llm_provider=llm_provider,
    )

    # ========================================================
    # CACHE
    # ========================================================

    cache_provider = RedisCacheProvider(
        redis_url=settings.redis_url,
        key_prefix=settings.redis_cache_key_prefix,
    )

    # ========================================================
    # GUARDRAILS
    # ========================================================

    authorization_guard = AuthorizationGuard()

    tenant_isolation_guard = TenantIsolationGuard()

    prompt_injection_guard = PromptInjectionGuard()

    pii_analyzer = RegexPIIAnalyzer()

    # ========================================================
    # RERANKER
    # ========================================================

    reranker = FastEmbedReranker()

    # ========================================================
    # RETRIEVAL VALIDATION
    # ========================================================

    retrieval_validator = BaselineRetrievalValidator()

    grounding_validator = BaselineGroundingValidator()

    # ========================================================
    # LANGGRAPH
    # ========================================================

    query_graph = create_query_graph(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
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

    query_service = QueryService(
        query_graph=query_graph,
        shutdown_resources=[
            llm_provider,
            cache_provider,
            vector_store,
        ],
    )

    return query_service