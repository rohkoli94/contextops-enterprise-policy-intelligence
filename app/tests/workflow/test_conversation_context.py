from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.config.settings import settings
from app.domain.conversation import (
    ConversationContext,
    ConversationMessage,
    ConversationRole,
)
from app.rag.workflow.nodes.conversation_context import (
    create_conversation_context_node,
)


class FakeConversationMemory:
    """
    Test-only fake.

    This is deliberately not part of the production application.
    """

    def __init__(self) -> None:
        self.received_conversation_id: str | None = None
        self.received_tenant_id: str | None = None
        self.received_limit: int | None = None

    async def get_context(
        self,
        conversation_id: str,
        tenant_id: str,
        *,
        recent_message_limit: int,
    ) -> ConversationContext:

        self.received_conversation_id = conversation_id
        self.received_tenant_id = tenant_id
        self.received_limit = recent_message_limit

        return ConversationContext(
            recent_messages=[
                ConversationMessage(
                    message_id=str(uuid4()),
                    conversation_id=conversation_id,
                    role=ConversationRole.USER,
                    content="What is the notice period?",
                    created_at=datetime.now(timezone.utc),
                ),
                ConversationMessage(
                    message_id=str(uuid4()),
                    conversation_id=conversation_id,
                    role=ConversationRole.ASSISTANT,
                    content="The policy states three months.",
                    created_at=datetime.now(timezone.utc),
                ),
            ],
            summary="User is discussing employee notice policies.",
        )


@pytest.mark.asyncio
async def test_node_loads_bounded_conversation_context() -> None:
    memory = FakeConversationMemory()

    node = create_conversation_context_node(memory)

    conversation_id = str(uuid4())

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "conversation_id": conversation_id,
    }

    result = await node(state)

    assert len(result["recent_messages"]) == 2

    assert (
        result["recent_messages"][0]["content"]
        == "What is the notice period?"
    )

    assert (
        result["recent_messages"][1]["content"]
        == "The policy states three months."
    )

    assert (
        result["conversation_summary"]
        == "User is discussing employee notice policies."
    )


@pytest.mark.asyncio
async def test_node_uses_configured_recent_message_limit() -> None:
    memory = FakeConversationMemory()

    node = create_conversation_context_node(memory)

    state = {
        "query": "What about managers?",
        "tenant_id": "tenant-001",
        "conversation_id": str(uuid4()),
    }

    await node(state)

    assert (
        memory.received_limit
        == settings.conversation_recent_message_limit
    )


@pytest.mark.asyncio
async def test_node_handles_new_conversation() -> None:
    memory = FakeConversationMemory()

    node = create_conversation_context_node(memory)

    state = {
        "query": "What is the leave policy?",
        "tenant_id": "tenant-001",
        "conversation_id": None,
    }

    result = await node(state)

    assert result["recent_messages"] == []
    assert result["conversation_summary"] is None

    # Memory must not be queried for a new conversation.
    assert memory.received_conversation_id is None