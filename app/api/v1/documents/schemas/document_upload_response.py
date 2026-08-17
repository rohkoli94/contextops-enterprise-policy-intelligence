import uuid

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    document_name: str
    version: int
    status: str