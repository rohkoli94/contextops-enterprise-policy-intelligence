from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.domain.conversation import (
    ConversationContext,
    ConversationMessage,
    ConversationRole,
)
from app.services.conversation_memory import (
    PostgresConversationMemory,
)


class FakeConversationRepository:
    """
    Test double only.

    This is deliberately located inside the test module and is
    NOT part of the production application architecture.
    """

    def __init__(self) -> None:
        self.context = ConversationContext(
            recent_messages=[],
            summary=None,
        )

        self.saved_messages: list[ConversationMessage] = []
        self.summary_updates: list[tuple[str, int]] = []

    async def get_context(
        self,
        conversation_id: UUID,
        tenant_id: str,
        recent_message_limit: int,
    ) -> ConversationContext:
        return self.context

    async def save_message(
        self,
        message: ConversationMessage,
        tenant_id: str,
    ) -> ConversationMessage:
        self.saved_messages.append(message)
        return message

    async def update_summary(
        self,
        conversation_id: UUID,
        tenant_id: str,
        summary: str,
        message_boundary: int,
    ) -> None:
        self.summary_updates.append(
            (summary, message_boundary)
        )


@pytest.mark.asyncio
async def test_get_context_delegates_to_repository() -> None:
    repository = FakeConversationRepository()
    memory = PostgresConversationMemory(repository)

    conversation_id = str(uuid4())

    context = await memory.get_context(
        conversation_id=conversation_id,
        tenant_id="tenant-001",
        recent_message_limit=6,
    )

    assert context.recent_messages == []
    assert context.summary is None


@pytest.mark.asyncio
async def test_get_context_rejects_invalid_conversation_id() -> None:
    repository = FakeConversationRepository()
    memory = PostgresConversationMemory(repository)

    with pytest.raises(ValueError, match="valid UUID"):
        await memory.get_context(
            conversation_id="not-a-uuid",
            tenant_id="tenant-001",
            recent_message_limit=6,
        )


@pytest.mark.asyncio
async def test_get_context_rejects_empty_tenant() -> None:
    repository = FakeConversationRepository()
    memory = PostgresConversationMemory(repository)

    with pytest.raises(
        ValueError,
        match="tenant_id cannot be empty",
    ):
        await memory.get_context(
            conversation_id=str(uuid4()),
            tenant_id="",
            recent_message_limit=6,
        )


@pytest.mark.asyncio
async def test_save_message_delegates_to_repository() -> None:
    repository = FakeConversationRepository()
    memory = PostgresConversationMemory(repository)

    message = ConversationMessage(
        message_id=str(uuid4()),
        conversation_id=str(uuid4()),
        role=ConversationRole.USER,
        content="What is the notice period?",
        created_at=datetime.now(timezone.utc),
    )

    result = await memory.save_message(
        message=message,
        tenant_id="tenant-001",
    )

    assert result == message
    assert repository.saved_messages == [message]


@pytest.mark.asyncio
async def test_update_summary_delegates_to_repository() -> None:
    repository = FakeConversationRepository()
    memory = PostgresConversationMemory(repository)

    conversation_id = str(uuid4())

    await memory.update_summary(
        conversation_id=conversation_id,
        tenant_id="tenant-001",
        summary="User is discussing leave policy.",
        message_boundary=12,
    )

    assert repository.summary_updates == [
        ("User is discussing leave policy.", 12)
    ]