from app.rag.workflow.edges import (
    route_after_retrieval_recovery,
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


def test_routes_to_contextops_after_successful_recovery() -> None:
    state = {
        "recovery_success": True,
    }

    assert (
        route_after_retrieval_recovery(state)
        == "contextops"
    )


def test_routes_to_response_after_failed_recovery() -> None:
    state = {
        "recovery_success": False,
    }

    assert (
        route_after_retrieval_recovery(state)
        == "response"
    )


def test_missing_recovery_result_defaults_to_response() -> None:
    state = {}

    assert (
        route_after_retrieval_recovery(state)
        == "response"
    )