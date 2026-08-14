from fastapi import APIRouter
from app.api.v1.query.schemas.query_response import QueryResponse
from app.api.v1.query.schemas.query_request import QueryRequest
from app.core.logging import get_logger
from app.providers.llm.microsoft_foundry import MicrosoftFoundryProvider
from app.services.query_service import QueryService

router = APIRouter()
logger = get_logger(__name__)

llm_provider = MicrosoftFoundryProvider()
query_service = QueryService(llm_provider)

@router.post("/query", response_model=QueryResponse)
def query_policy(request: QueryRequest) -> QueryResponse:
    logger.info("Policy query request received")

    response = query_service.ask(request.query)
    
    return QueryResponse(
        answer=response.content,
        status="success"
    )