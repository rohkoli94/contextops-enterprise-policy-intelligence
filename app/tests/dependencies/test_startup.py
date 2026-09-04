from types import SimpleNamespace

from app.dependencies import startup


class FakeQueryGraph:
    pass


class FakeQueryService:
    def __init__(self) -> None:
        self.query_graph = FakeQueryGraph()


def test_initialize_application_stores_shared_query_service(
    monkeypatch,
) -> None:
    fake_query_service = FakeQueryService()

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

    app = SimpleNamespace(
        state=SimpleNamespace(),
    )

    startup.initialize_application(app)

    assert (
        app.state.query_service
        is fake_query_service
    )

    assert (
        app.state.query_graph
        is fake_query_service.query_graph
    )