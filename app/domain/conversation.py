from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class ConversationMessage:
    """
    Immutable domain representation of a conversation message.
    """

    message_id: str
    conversation_id: str
    role: ConversationRole
    content: str
    created_at: datetime


@dataclass(frozen=True)
class ConversationContext:
    """
    Bounded conversation context supplied to the query workflow.

    recent_messages contains only a configurable recent window.
    summary represents older conversation history.
    """

    recent_messages: list[ConversationMessage]
    summary: str | None = None