from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        strip_whitespace=True,
        description="User's query about enterprise policy"
    )