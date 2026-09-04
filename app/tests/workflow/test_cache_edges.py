from app.rag.workflow.edges import (
    route_after_cache_lookup,
)


def test_cache_hit_routes_to_cached_response() -> None:
    state = {
        "cache_hit": True,
    }

    assert (
        route_after_cache_lookup(state)
        == "cached_response"
    )


def test_cache_miss_routes_to_retrieval() -> None:
    state = {
        "cache_hit": False,
    }

    assert (
        route_after_cache_lookup(state)
        == "hybrid_retrieval"
    )


def test_missing_cache_result_routes_to_retrieval() -> None:
    state = {}

    assert (
        route_after_cache_lookup(state)
        == "hybrid_retrieval"
    )