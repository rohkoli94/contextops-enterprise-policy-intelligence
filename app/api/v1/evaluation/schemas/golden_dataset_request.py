from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenDatasetRequest(BaseModel):
    trace_ids: list[str] = Field(
        min_length=1,
        description=(
            "Langfuse trace IDs to convert into "
            "golden-dataset candidates."
        ),
    )
