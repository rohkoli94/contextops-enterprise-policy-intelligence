from pydantic import BaseModel, Field

from app.api.v1.query.schemas.query_filter import (
    QueryFilter,
)


class QueryRequest(BaseModel):
    """
    API request for an enterprise policy query.
    """

    query: str = Field(
        min_length=1,
    )

    tenant_id: str = Field(
        min_length=1,
    )

    conversation_id: str | None = None

    filters: QueryFilter | None = None