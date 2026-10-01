import uuid

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    document_name: str
    version: int
    active_version: int | None = None
    status: str
