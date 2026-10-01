import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentListItem(BaseModel):
    document_id: uuid.UUID
    document_name: str
    current_version: int
    active_version: int | None = None
    status: str
    categories: list[str]
    tags: list[str]
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
