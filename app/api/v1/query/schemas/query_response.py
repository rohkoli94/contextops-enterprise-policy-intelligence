from typing import Any

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """
    API response for an enterprise policy query.
    """

    answer: str

    status: str

    citations: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )