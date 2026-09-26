from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.evaluation.schemas.generate_golden_files_request import (
    GenerateGoldenFilesRequest,
)
from app.api.v1.evaluation.schemas.generate_golden_files_response import (
    GenerateGoldenFilesResponse,
)
from app.api.v1.evaluation.schemas.golden_dataset_request import (
    GoldenDatasetRequest,
)
from app.api.v1.evaluation.schemas.golden_dataset_response import (
    GoldenDatasetResponse,
)
from app.evaluation.golden_dataset_file_service import (
    GoldenDatasetFileService,
)
from app.evaluation.langfuse_golden_dataset_service import (
    LangfuseGoldenDatasetService,
)


router = APIRouter(
    prefix="/evaluation",
    tags=["evaluation"],
)


@router.post(
    "/golden-dataset",
    response_model=GoldenDatasetResponse,
)
def generate_golden_dataset(
    request: GoldenDatasetRequest,
) -> GoldenDatasetResponse:
    service = LangfuseGoldenDatasetService()

    examples = service.build_from_traces(
        trace_ids=request.trace_ids,
    )

    return GoldenDatasetResponse(
        examples=examples,
    )


@router.post(
    "/golden-dataset/generate-files",
    response_model=GenerateGoldenFilesResponse,
)
async def generate_golden_dataset_files(
    request: GenerateGoldenFilesRequest,
) -> GenerateGoldenFilesResponse:
    service = GoldenDatasetFileService()

    result = await service.generate(
        trace_ids=request.trace_ids,
    )

    return GenerateGoldenFilesResponse(
        **result,
    )