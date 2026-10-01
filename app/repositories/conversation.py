from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.domain.conversation import (
    ConversationContext,
    ConversationMessage,
    ConversationRole,
)
from app.models.conversation import Conversation
from app.models.conversation_message import (
    ConversationMessage as ConversationMessageModel,
)
from app.models.conversation_summary import (
    ConversationSummary,
)


class ConversationRepository:
    """
    PostgreSQL repository for conversation persistence.

    The repository is application-scoped, but SQLAlchemy sessions
    are NOT shared.

    A new AsyncSession is created for each repository operation.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self.session_factory = session_factory

    # ========================================================
    # CONVERSATION
    # ========================================================

    async def get_conversation(
        self,
        conversation_id: UUID,
        tenant_id: str,
    ) -> Conversation | None:

        async with self.session_factory() as db:

            statement = (
                select(Conversation)
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            result = await db.execute(
                statement
            )

            return result.scalar_one_or_none()

    async def create_conversation(
        self,
        tenant_id: str,
        title: str | None = None,
        conversation_id: UUID | None = None,
    ) -> Conversation:

        async with self.session_factory() as db:

            conversation_kwargs = {
                "tenant_id": tenant_id,
                "title": title,
            }

            if conversation_id is not None:
                conversation_kwargs[
                    "conversation_id"
                ] = conversation_id

            conversation = Conversation(
                **conversation_kwargs,
            )

            db.add(conversation)

            await db.commit()
            await db.refresh(
                conversation
            )

            return conversation

    # ========================================================
    # MESSAGES
    # ========================================================

    async def get_recent_messages(
        self,
        conversation_id: UUID,
        tenant_id: str,
        limit: int,
    ) -> list[ConversationMessage]:

        if limit < 0:
            raise ValueError(
                "Message limit cannot be negative."
            )

        if limit == 0:
            return []

        async with self.session_factory() as db:

            conversation_statement = (
                select(
                    Conversation.conversation_id
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            conversation_result = (
                await db.execute(
                    conversation_statement
                )
            )

            conversation_exists = (
                conversation_result.scalar_one_or_none()
                is not None
            )

            if not conversation_exists:
                return []

            statement = (
                select(
                    ConversationMessageModel
                )
                .where(
                    ConversationMessageModel.conversation_id
                    == conversation_id
                )
                .order_by(
                    ConversationMessageModel.created_at.desc()
                )
                .limit(limit)
            )

            result = await db.execute(
                statement
            )

            rows = list(
                result.scalars().all()
            )

            rows.reverse()

            return [
                self._to_domain_message(row)
                for row in rows
            ]

    async def save_message(
        self,
        message: ConversationMessage,
        tenant_id: str,
    ) -> ConversationMessage:

        conversation_id = UUID(
            message.conversation_id
        )

        async with self.session_factory() as db:

            conversation_statement = (
                select(
                    Conversation.conversation_id
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            conversation_result = (
                await db.execute(
                    conversation_statement
                )
            )

            conversation_exists = (
                conversation_result.scalar_one_or_none()
                is not None
            )

            if not conversation_exists:
                raise ValueError(
                    "Conversation does not exist or "
                    "does not belong to the tenant."
                )

            message_model = (
                ConversationMessageModel(
                    conversation_id=conversation_id,
                    role=message.role.value,
                    content=message.content,
                    created_at=message.created_at,
                )
            )

            db.add(message_model)

            await db.commit()

            await db.refresh(
                message_model
            )

            return self._to_domain_message(
                message_model
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    async def get_latest_summary(
        self,
        conversation_id: UUID,
        tenant_id: str,
    ) -> str | None:

        async with self.session_factory() as db:

            conversation_statement = (
                select(
                    Conversation.conversation_id
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            conversation_result = (
                await db.execute(
                    conversation_statement
                )
            )

            conversation_exists = (
                conversation_result.scalar_one_or_none()
                is not None
            )

            if not conversation_exists:
                return None

            statement = (
                select(
                    ConversationSummary
                )
                .where(
                    ConversationSummary.conversation_id
                    == conversation_id
                )
                .order_by(
                    ConversationSummary.updated_at.desc(),
                    ConversationSummary.created_at.desc(),
                )
                .limit(1)
            )

            result = await db.execute(
                statement
            )

            summary = (
                result.scalar_one_or_none()
            )

            if summary is None:
                return None

            return summary.summary

    async def update_summary(
        self,
        conversation_id: UUID,
        tenant_id: str,
        summary: str,
        message_boundary: int,
    ) -> None:

        async with self.session_factory() as db:

            conversation_statement = (
                select(
                    Conversation.conversation_id
                )
                .where(
                    Conversation.conversation_id
                    == conversation_id,
                    Conversation.tenant_id
                    == tenant_id,
                )
            )

            conversation_result = (
                await db.execute(
                    conversation_statement
                )
            )

            conversation_exists = (
                conversation_result.scalar_one_or_none()
                is not None
            )

            if not conversation_exists:
                raise ValueError(
                    "Conversation does not exist or "
                    "does not belong to the tenant."
                )

            statement = (
                select(
                    ConversationSummary
                )
                .where(
                    ConversationSummary.conversation_id
                    == conversation_id
                )
                .order_by(
                    ConversationSummary.updated_at.desc(),
                    ConversationSummary.created_at.desc(),
                )
                .limit(1)
            )

            result = await db.execute(
                statement
            )

            existing = (
                result.scalar_one_or_none()
            )

            if existing is None:

                summary_model = (
                    ConversationSummary(
                        conversation_id=conversation_id,
                        summary=summary,
                        message_boundary=message_boundary,
                    )
                )

                db.add(summary_model)

            else:

                existing.summary = summary
                existing.message_boundary = (
                    message_boundary
                )

            await db.commit()

    # ========================================================
    # BOUNDED CONTEXT
    # ========================================================

    async def get_context(
        self,
        conversation_id: UUID,
        tenant_id: str,
        recent_message_limit: int,
    ) -> ConversationContext:

        messages = (
            await self.get_recent_messages(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                limit=recent_message_limit,
            )
        )

        summary = (
            await self.get_latest_summary(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
            )
        )

        return ConversationContext(
            recent_messages=messages,
            summary=summary,
        )

    # ========================================================
    # MAPPING
    # ========================================================

    @staticmethod
    def _to_domain_message(
        message: ConversationMessageModel,
    ) -> ConversationMessage:

        return ConversationMessage(
            message_id=str(
                message.message_id
            ),
            conversation_id=str(
                message.conversation_id
            ),
            role=ConversationRole(
                message.role
            ),
            content=message.content,
            created_at=message.created_at,
        )
