from pydantic import BaseModel, ConfigDict


class QueryFilter(BaseModel):
    """
    Optional metadata filters for retrieval.

    Multiple values are treated as OR within the same field.

    Different fields are combined as AND conditions.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    document_id: str | None = None

    document_version_id: str | None = None

    categories: list[str] | None = None

    tags: list[str] | None = None

    content_type: str | None = None