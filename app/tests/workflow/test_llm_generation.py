import pytest

from app.providers.llm.base import (
    LLMResponse,
)
from app.rag.workflow.nodes.llm_generation import (
    create_llm_generation_node,
)


class FakeLLMProvider:
    def __init__(self) -> None:
        self.received_request = None

    async def agenerate(self, request):
        self.received_request = request

        return LLMResponse(
            content="The notice period is three months.",
            model="test-model",
            provider="test-provider",
        )


@pytest.mark.asyncio
async def test_llm_generation_uses_original_query() -> None:
    provider = FakeLLMProvider()

    node = create_llm_generation_node(
        provider
    )

    state = {
        "query": "What about managers?",
        "contextualized_query": (
            "What is the notice period policy for managers?"
        ),
        "tenant_id": "tenant-001",
        "context": (
            "[SOURCE 1]\n"
            "Managers have a three-month notice period."
        ),
    }

    result = await node(state)

    assert (
        provider.received_request.user_prompt
        == "What about managers?"
    )

    assert (
        provider.received_request.context
        == (
            "[SOURCE 1]\n"
            "Managers have a three-month notice period."
        )
    )

    assert (
        result["answer"]
        == "The notice period is three months."
    )


@pytest.mark.asyncio
async def test_llm_generation_returns_answer() -> None:
    provider = FakeLLMProvider()

    node = create_llm_generation_node(
        provider
    )

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "context": "Employees receive annual leave.",
    }

    result = await node(state)

    assert result["answer"] == (
        "The notice period is three months."
    )