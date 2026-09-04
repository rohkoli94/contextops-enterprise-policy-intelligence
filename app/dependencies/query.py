from fastapi import Request

from app.services.query_service import QueryService


def get_query_service(
    request: Request,
) -> QueryService:
    """
    Return the application-scoped QueryService.

    QueryService is created once during FastAPI startup
    and stored in app.state.

    No QueryService or provider is created per request.
    """

    return request.app.state.query_service