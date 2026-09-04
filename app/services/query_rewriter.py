from abc import ABC, abstractmethod


class QueryRewriter(ABC):
    """
    Application-level abstraction for converting a conversational
    user query into a standalone retrieval query.

    Example:

        Previous:
            "What is the notice period for employees?"

        Current:
            "What about managers?"

        Retrieval query:
            "What is the notice period policy for managers?"
    """

    @abstractmethod
    async def rewrite(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        recent_messages: list[dict[str, object]],
    ) -> str:
        raise NotImplementedError