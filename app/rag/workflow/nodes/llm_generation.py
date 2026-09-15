from collections.abc import Awaitable, Callable

from app.prompts.llm_generation import LLM_GENERATION_SYSTEM_PROMPT
from app.providers.llm.base import (
    LLMProvider,
    LLMRequest,
)
from app.rag.workflow.state import QueryState


def create_llm_generation_node(
    llm_provider: LLMProvider,
) -> Callable[[QueryState], Awaitable[QueryState]]:
    """
    Create the LangGraph LLM generation node.

    The LLM receives:
        - the original user question
        - the ContextOps-built evidence context

    The contextualized query is used for retrieval only.
    """

    async def node(
        state: QueryState,
    ) -> QueryState:

        question = state["query"]
        context = state.get("context", "")

        request = LLMRequest(
            system_prompt=LLM_GENERATION_SYSTEM_PROMPT,
            user_prompt=question,
            context=context,
        )

        response = await llm_provider.agenerate(
            request
        )

        return {
            **state,
            "answer": response.content,
        }

    return node