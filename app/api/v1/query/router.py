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
        conversation_id=result.get(
            "conversation_id",
        ),
        citations=result.get(
            "citations",
            [],
        ),
        metadata=result.get(
            "final_response_metadata",
            {},
        ),
    )


# logger.info("") 
# logger.info("Query service result: %s",result) 
# logger.info("") 

#  logger.info(
#         "Query service result: %s",
#         result,
#     )

#     logger.info("---------------")

#     logger.info(
#     "Query completed successfully | "
#     "conversation_id=%s | "
#     "citations=%d | "
#     "answer=%s",
#     result.get("conversation_id"),
#     len(result.get("citations", [])),
#     result.get("answer"),
#     ) 