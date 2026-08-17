from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.category import Category, document_categories
from app.models.tag import Tag, document_tags


"""
rohit notes:
This ensures the models are imported and registered with SQLAlchemy's Base.metadata.

Why is this important?

Alembic needs to see:

Base.metadata
    │
    ├── documents
    ├── document_versions
    ├── categories
    ├── document_categories
    ├── tags
    └── document_tags

Without importing the model modules, Base.metadata may not contain all your tables.

"""