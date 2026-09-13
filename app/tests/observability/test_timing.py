import asyncio

import pytest

from app.observability.timing import (
    create_timed_node,
    record_total_query_time,
)


@pytest.mark.asyncio
async def test_timed_node_records_stage_duration() -> None:
    async def fake_node(state):
        await asyncio.sleep(0)
        return {
            **state,
            "answer": "done",
        }

    timed_node = create_timed_node(
        name="llm_generation",
        node=fake_node,
    )

    result = await timed_node(
        {
            "timings": {
                "stages": {},
                "total_ms": None,
            }
        }
    )

    timings = result["timings"]

    assert "llm_generation" in (
        timings["stages"]
    )

    assert isinstance(
        timings["stages"]["llm_generation"],
        float,
    )

    assert timings["total_ms"] is None


@pytest.mark.asyncio
async def test_timed_node_preserves_existing_stage_timings() -> None:
    async def fake_node(state):
        return {
            **state,
            "value": 10,
        }

    timed_node = create_timed_node(
        name="retrieval",
        node=fake_node,
    )

    result = await timed_node(
        {
            "timings": {
                "stages": {
                    "input_validation": 1.5,
                },
                "total_ms": None,
            }
        }
    )

    assert (
        result["timings"]["stages"][
            "input_validation"
        ]
        == 1.5
    )

    assert (
        "retrieval"
        in result["timings"]["stages"]
    )


@pytest.mark.asyncio
async def test_timed_node_records_duration_on_failure() -> None:
    async def failing_node(state):
        raise RuntimeError(
            "node failed"
        )

    timed_node = create_timed_node(
        name="retrieval",
        node=failing_node,
    )

    state = {
        "timings": {
            "stages": {},
            "total_ms": None,
        }
    }

    with pytest.raises(
        RuntimeError,
        match="node failed",
    ):
        await timed_node(state)

    assert (
        "retrieval"
        in state["timings"]["stages"]
    )

    assert isinstance(
        state["timings"]["stages"][
            "retrieval"
        ],
        float,
    )


def test_record_total_query_time() -> None:
    state = {
        "answer": "done",
        "timings": {
            "stages": {
                "retrieval": 25.2,
            },
            "total_ms": None,
        },
    }

    result = record_total_query_time(
        state,
        125.6789,
    )

    assert (
        result["timings"]["total_ms"]
        == 125.679
    )

    assert (
        result["timings"]["stages"][
            "retrieval"
        ]
        == 25.2
    )


def test_timed_node_rejects_empty_name() -> None:
    async def fake_node(state):
        return state

    with pytest.raises(
        ValueError,
        match="Timing node name cannot be empty",
    ):
        create_timed_node(
            name="   ",
            node=fake_node,
        )