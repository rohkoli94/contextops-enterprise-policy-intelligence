from pathlib import Path

from app.guardrails import (
    AuthorizationGuard,
    PromptInjectionGuard,
    RegexPIIAnalyzer,
    TenantIsolationGuard,
)
from app.rag.workflow.graph import create_query_graph


def build_graph():
    """
    Build the ContextOps query graph with lightweight
    dependencies for visualization only.
    """

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

    class FakeReranker:
        pass

    class FakeRetrievalValidator:
        pass

    class FakeGroundingValidator:
        pass

    return create_query_graph(
        llm_provider=FakeLLMProvider(),
        hybrid_retriever=FakeHybridRetriever(),
        conversation_memory=FakeConversationMemory(),
        query_rewriter=FakeQueryRewriter(),
        cache_provider=FakeCacheProvider(),
        authorization_guard=AuthorizationGuard(),
        tenant_isolation_guard=TenantIsolationGuard(),
        prompt_injection_guard=PromptInjectionGuard(),
        pii_analyzer=RegexPIIAnalyzer(),
        reranker=FakeReranker(),
        retrieval_validator=FakeRetrievalValidator(),
        grounding_validator=FakeGroundingValidator(),
    )


def main() -> None:
    graph = build_graph()

    output_path = (
        Path(__file__).resolve().parent
        / "contextops_query_graph.png"
    )

    image = graph.get_graph().draw_mermaid_png()

    output_path.write_bytes(image)

    print(
        f"Graph image created:\n{output_path}"
    )


if __name__ == "__main__":
    main()