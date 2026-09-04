from fastapi import FastAPI

from app.dependencies import startup


def test_application_startup_composes_shared_query_service(
    monkeypatch,
) -> None:
    app = FastAPI()

    class FakeQueryGraph:
        pass

    class FakeQueryService:
        def __init__(self) -> None:
            self.query_graph = FakeQueryGraph()

    fake_query_service = FakeQueryService()

    # Do not initialize real Qdrant/Azure infrastructure.
    monkeypatch.setattr(
        startup,
        "initialize_rag",
        lambda: None,
    )

    monkeypatch.setattr(
        startup,
        "create_query_service",
        lambda: fake_query_service,
    )

    startup.initialize_application(app)

    assert app.state.query_service is fake_query_service
    assert (
        app.state.query_graph
        is fake_query_service.query_graph
    )