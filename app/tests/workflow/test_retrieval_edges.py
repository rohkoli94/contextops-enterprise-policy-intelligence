from app.rag.workflow.edges import (
    route_after_retrieval_validation,
)


def test_routes_to_contextops_when_retrieval_is_sufficient() -> None:
    state = {
        "retrieval_sufficient": True,
    }

    assert (
        route_after_retrieval_validation(state)
        == "contextops"
    )


def test_routes_to_recovery_when_retrieval_is_insufficient() -> None:
    state = {
        "retrieval_sufficient": False,
    }

    assert (
        route_after_retrieval_validation(state)
        == "retrieval_recovery"
    )


def test_missing_validation_result_defaults_to_recovery() -> None:
    state = {}

    assert (
        route_after_retrieval_validation(state)
        == "retrieval_recovery"
    )