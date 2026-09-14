from typing import Annotated
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    UploadFile,
)

from app.api.v1.documents.schemas.document_upload_response import (
    DocumentUploadResponse,
)

from app.dependencies.document import (
    get_async_document_service,
)

from app.services.document_service import (
    DocumentService,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=202,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(...),
    ],
    document_name: Annotated[
        str,
        Form(...),
    ],
    categories: Annotated[
        list[str] | None,
        Form(),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Form(),
    ] = None,
    document_service: DocumentService = Depends(
        get_async_document_service
    ),
) -> DocumentUploadResponse:

    return await document_service.aupload_document(
        stream=file.file,
        file_name=file.filename or "document",
        content_type=(
            file.content_type
            or "application/octet-stream"
        ),
        document_name=document_name,
        categories=categories,
        tags=tags,
        background_tasks=background_tasks,
    )


@router.post(
    "/{document_id}/versions",
    response_model=DocumentUploadResponse,
    status_code=202,
)
async def upload_new_document_version(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(...),
    ],
    document_service: DocumentService = Depends(
        get_async_document_service
    ),
) -> DocumentUploadResponse:

    return await document_service.aupload_new_version(
        document_id=document_id,
        stream=file.file,
        file_name=file.filename or "document",
        content_type=(
            file.content_type
            or "application/octet-stream"
        ),
        background_tasks=background_tasks,
    )