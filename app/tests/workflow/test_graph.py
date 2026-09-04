from app.guardrails import (
    AuthorizationGuard,
    PromptInjectionGuard,
    RegexPIIAnalyzer,
    TenantIsolationGuard,
)
from app.rag.workflow.graph import create_query_graph


class FakeLLMProvider:
    pass


class FakeHybridRetriever:
    pass


class FakeConversationMemory:
    pass


class FakeQueryRewriter:
    pass


class FakeCacheProvider:
    pass


def test_query_graph_compiles() -> None:
    graph = create_query_graph(
        llm_provider=FakeLLMProvider(),
        hybrid_retriever=FakeHybridRetriever(),
        conversation_memory=FakeConversationMemory(),
        query_rewriter=FakeQueryRewriter(),
        cache_provider=FakeCacheProvider(),
        authorization_guard=AuthorizationGuard(),
        tenant_isolation_guard=TenantIsolationGuard(),
        prompt_injection_guard=PromptInjectionGuard(),
        pii_analyzer=RegexPIIAnalyzer(),
    )

    assert graph is not None