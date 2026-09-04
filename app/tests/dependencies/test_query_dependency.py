from types import SimpleNamespace

from app.dependencies.query import (
    get_query_service,
)


def test_get_query_service_returns_application_scoped_service() -> None:
    query_service = object()

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                query_service=query_service,
            ),
        ),
    )

    result = get_query_service(request)

    assert result is query_service