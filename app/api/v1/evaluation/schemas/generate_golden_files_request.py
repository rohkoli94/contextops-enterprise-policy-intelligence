from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateGoldenFilesRequest(BaseModel):
    trace_ids: list[str] = Field(
        min_length=1,
        description=(
            "Langfuse trace IDs used to generate the "
            "golden dataset and source candidates."
        ),
    )
