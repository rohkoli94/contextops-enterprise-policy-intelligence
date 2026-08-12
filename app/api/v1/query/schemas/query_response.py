from pydantic import BaseModel, Field

class QueryResponse(BaseModel):
    answer: str
    status: str