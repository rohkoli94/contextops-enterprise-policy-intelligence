from app.prompts.query_contextualization import (
    QUERY_CONTEXTUALIZATION_SYSTEM_PROMPT,
)
from app.providers.llm.base import LLMProvider, LLMRequest
from app.services.query_rewriter import QueryRewriter


class MicrosoftFoundryQueryRewriter(QueryRewriter):
    """
    Microsoft Foundry implementation of QueryRewriter.

    The original user query is preserved in QueryState.
    This component only produces the standalone query used
    for retrieval.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self.llm_provider = llm_provider

    async def rewrite(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        recent_messages: list[dict[str, object]],
    ) -> str:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        summary = (
            conversation_summary.strip()
            if conversation_summary
            else "No conversation summary is available."
        )

        recent_context = self._format_recent_messages(
            recent_messages
        )

        user_prompt = f"""
Conversation summary:
{summary}

Recent conversation:
{recent_context}

Current user query:
{query}

Standalone retrieval query:
""".strip()

        response = await self.llm_provider.agenerate(
            LLMRequest(
                system_prompt=QUERY_CONTEXTUALIZATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        )

        rewritten_query = response.content.strip()

        # Safe fallback: retrieval can continue using the
        # original query if contextualization returns nothing.
        return rewritten_query or query.strip()

    @staticmethod
    def _format_recent_messages(
        recent_messages: list[dict[str, object]],
    ) -> str:
        if not recent_messages:
            return "No recent messages are available."

        lines: list[str] = []

        for message in recent_messages:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).strip()

            if content:
                lines.append(f"{role}: {content}")

        return (
            "\n".join(lines)
            if lines
            else "No recent messages are available."
        )