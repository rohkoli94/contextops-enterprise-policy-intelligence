from fastapi import APIRouter, Depends

from app.api.v1.query.schemas.query_request import (
    QueryRequest,
)
from app.api.v1.query.schemas.query_response import (
    QueryResponse,
)
from app.core.logging import get_logger
from app.dependencies.query import (
    get_query_service,
)
from app.services.query_service import (
    QueryService,
)


router = APIRouter()

logger = get_logger(__name__)


# ============================================================
# QUERY API
# ============================================================

@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query_policy(
    request: QueryRequest,
    query_service: QueryService = Depends(
        get_query_service,
    ),
) -> QueryResponse:
    """
    Execute an enterprise policy query.

    Flow:

        HTTP Request
            ↓
        FastAPI Dependency
            ↓
        app.state.query_service
            ↓
        QueryService
            ↓
        LangGraph
            ↓
        QueryResponse
    """

    logger.info(
        "User query request received",
    )

    result = await query_service.ask(
        question=request.query,
        tenant_id=request.tenant_id,
        conversation_id=request.conversation_id,
        filters=request.filters,
    )

    return QueryResponse(
        answer=result["answer"],
        status="success",
        citations=result.get(
            "citations",
            [],
        ),
        metadata=result.get(
            "final_response_metadata",
            {},
        ),
    )