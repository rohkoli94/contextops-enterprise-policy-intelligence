from pydantic import BaseModel, Field

from app.api.v1.query.schemas.query_filter import (
    QueryFilter,
)


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
    )

    tenant_id: str = Field(
        min_length=1,
    )

    filters: QueryFilter | None = None