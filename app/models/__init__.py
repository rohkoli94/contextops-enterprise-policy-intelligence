from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.category import Category, document_categories
from app.models.tag import Tag, document_tags
from app.models.conversation import Conversation
from app.models.conversation_message import ConversationMessage
from app.models.conversation_summary import ConversationSummary


"""
rohit notes:
This ensures the models are imported and registered with SQLAlchemy's Base.metadata.

Why is this important?

Alembic needs to see all models through:

Base.metadata

Without importing the model modules, Alembic may not detect
the new conversation tables during autogenerate.
"""