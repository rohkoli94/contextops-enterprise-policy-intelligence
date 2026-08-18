from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.v1.documents.schemas.document_upload_response import (
    DocumentUploadResponse,
)

from app.dependencies.document import get_document_service
from app.services.document_service import DocumentService


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=201,
)
def upload_document(
    file: Annotated[UploadFile, File(...)],
    document_name: Annotated[str, Form(...)],
    categories: Annotated[list[str] | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:

    return document_service.upload_document(
        stream=file.file,
        file_name=file.filename or "document",
        content_type=file.content_type or "application/octet-stream",
        document_name=document_name,
        categories=categories,
        tags=tags,
    )


@router.post(
    "/{document_id}/versions",
    response_model=DocumentUploadResponse,
    status_code=201,
)
def upload_new_document_version(
    document_id: uuid.UUID,
    file: Annotated[UploadFile, File(...)],
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:

    return document_service.upload_new_version(
        document_id=document_id,
        stream=file.file,
        file_name=file.filename or "document",
        content_type=file.content_type or "application/octet-stream",
    )