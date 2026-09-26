from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.conversation import (
    ConversationContext,
    ConversationMessage,
)
from app.repositories.conversation import (
    ConversationRepository,
)


class ConversationMemory(ABC):
    """
    Application-level abstraction for conversation memory.

    Full conversation history is persisted in PostgreSQL.

    Only bounded context is exposed to the query workflow:

        rolling summary
              +
        recent N messages
    """

    @abstractmethod
    async def ensure_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
        title: str | None = None,
    ) -> str:
        """
        Ensure a conversation exists for the tenant.

        Creates it when it does not already exist.
        Returns the conversation ID.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_context(
        self,
        conversation_id: str,
        tenant_id: str,
        *,
        recent_message_limit: int,
    ) -> ConversationContext:
        """
        Retrieve bounded conversation context.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_message(
        self,
        message: ConversationMessage,
        tenant_id: str,
    ) -> ConversationMessage:
        """
        Persist a conversation message.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_summary(
        self,
        conversation_id: str,
        tenant_id: str,
        summary: str,
        message_boundary: int,
    ) -> None:
        """
        Create or update the rolling conversation summary.
        """
        raise NotImplementedError


class PostgresConversationMemory(ConversationMemory):
    """
    PostgreSQL-backed ConversationMemory implementation.
    """

    def __init__(
        self,
        repository: ConversationRepository,
    ) -> None:
        self.repository = repository

    # ========================================================
    # CONVERSATION
    # ========================================================

    async def ensure_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
        title: str | None = None,
    ) -> str:
        if not conversation_id or not conversation_id.strip():
            raise ValueError(
                "conversation_id cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "tenant_id cannot be empty."
            )

        try:
            conversation_uuid = UUID(
                conversation_id.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "conversation_id must be a valid UUID."
            ) from exc

        existing = await self.repository.get_conversation(
            conversation_id=conversation_uuid,
            tenant_id=tenant_id.strip(),
        )

        if existing is not None:
            return str(
                existing.conversation_id
            )

        conversation = (
            await self.repository.create_conversation(
                tenant_id=tenant_id.strip(),
                title=title,
                conversation_id=conversation_uuid,
            )
        )

        return str(
            conversation.conversation_id
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    async def get_context(
        self,
        conversation_id: str,
        tenant_id: str,
        *,
        recent_message_limit: int,
    ) -> ConversationContext:

        if not conversation_id or not conversation_id.strip():
            raise ValueError(
                "conversation_id cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "tenant_id cannot be empty."
            )

        if recent_message_limit < 0:
            raise ValueError(
                "recent_message_limit cannot be negative."
            )

        try:
            conversation_uuid = UUID(
                conversation_id
            )
        except ValueError as exc:
            raise ValueError(
                "conversation_id must be a valid UUID."
            ) from exc

        return await self.repository.get_context(
            conversation_id=conversation_uuid,
            tenant_id=tenant_id.strip(),
            recent_message_limit=recent_message_limit,
        )

    # ========================================================
    # MESSAGE
    # ========================================================

    async def save_message(
        self,
        message: ConversationMessage,
        tenant_id: str,
    ) -> ConversationMessage:

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "tenant_id cannot be empty."
            )

        if not message.conversation_id:
            raise ValueError(
                "message.conversation_id cannot be empty."
            )

        try:
            UUID(message.conversation_id)
        except ValueError as exc:
            raise ValueError(
                "message.conversation_id must be a valid UUID."
            ) from exc

        return await self.repository.save_message(
            message=message,
            tenant_id=tenant_id.strip(),
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    async def update_summary(
        self,
        conversation_id: str,
        tenant_id: str,
        summary: str,
        message_boundary: int,
    ) -> None:

        if not conversation_id or not conversation_id.strip():
            raise ValueError(
                "conversation_id cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "tenant_id cannot be empty."
            )

        if not summary or not summary.strip():
            raise ValueError(
                "summary cannot be empty."
            )

        if message_boundary < 0:
            raise ValueError(
                "message_boundary cannot be negative."
            )

        try:
            conversation_uuid = UUID(
                conversation_id
            )
        except ValueError as exc:
            raise ValueError(
                "conversation_id must be a valid UUID."
            ) from exc

        await self.repository.update_summary(
            conversation_id=conversation_uuid,
            tenant_id=tenant_id.strip(),
            summary=summary.strip(),
            message_boundary=message_boundary,
        )