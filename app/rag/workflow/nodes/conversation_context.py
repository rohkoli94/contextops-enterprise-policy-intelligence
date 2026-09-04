from app.config.settings import settings
from app.rag.workflow.state import QueryState
from app.services.conversation_memory import ConversationMemory


def create_conversation_context_node(
    conversation_memory: ConversationMemory,
):
    """
    Create the LangGraph node responsible for loading bounded
    conversation context.

    Full conversation history remains in PostgreSQL.

    Only:
        - recent N messages
        - rolling summary

    are placed into QueryState.
    """

    async def node(state: QueryState) -> QueryState:
        conversation_id = state.get("conversation_id")
        tenant_id = state["tenant_id"]

        # --------------------------------------------------
        # New conversation
        # --------------------------------------------------
        if not conversation_id:
            return {
                **state,
                "recent_messages": [],
                "conversation_summary": None,
            }

        # --------------------------------------------------
        # Existing conversation
        # --------------------------------------------------
        context = await conversation_memory.get_context(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            recent_message_limit=(
                settings.conversation_recent_message_limit
            ),
        )

        recent_messages = [
            {
                "message_id": message.message_id,
                "conversation_id": message.conversation_id,
                "role": message.role.value,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in context.recent_messages
        ]

        return {
            **state,
            "recent_messages": recent_messages,
            "conversation_summary": context.summary,
        }

    return node