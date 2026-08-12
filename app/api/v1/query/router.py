from fastapi import APIRouter
from app.api.v1.query.schemas.query_response import QueryResponse
from app.api.v1.query.schemas.query_request import QueryRequest
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/query", response_model=QueryResponse)
def query_policy(request: QueryRequest) -> QueryResponse:
    logger.info("Policy query request received")
    
    return QueryResponse(
        answer=f"Received question: {request.query}",
        status="recieved"
    )