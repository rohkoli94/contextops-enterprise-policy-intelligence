from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from app.rag.workflow.state import QueryState


def _get_timings(
    state: QueryState,
) -> dict[str, Any]:
    """
    Return the existing serializable timing structure.

    Every workflow node receives and returns a state dictionary,
    so timings are explicitly copied into the returned state
    instead of relying on an in-memory runtime object.
    """

    existing_timings = state.get(
        "timings"
    )

    if not isinstance(
        existing_timings,
        dict,
    ):
        return {
            "stages": {},
            "total_ms": None,
        }

    stages = existing_timings.get(
        "stages"
    )

    if not isinstance(
        stages,
        dict,
    ):
        stages = {}

    total_ms = existing_timings.get(
        "total_ms"
    )

    return {
        "stages": dict(stages),
        "total_ms": total_ms,
    }


def create_timed_node(
    *,
    name: str,
    node: Callable[
        [QueryState],
        Awaitable[QueryState],
    ],
) -> Callable[
    [QueryState],
    Awaitable[QueryState],
]:
    """
    Wrap an asynchronous LangGraph node and record execution time.

    The wrapped node remains responsible for all business logic.

    Timing information is propagated through QueryState so every
    subsequent node can retain timings recorded by earlier nodes.

    Timing failures never change application behavior.
    """

    if not name or not name.strip():
        raise ValueError(
            "Timing node name cannot be empty."
        )

    stage_name = name.strip()

    async def timed_node(
        state: QueryState,
    ) -> QueryState:
        started_at = perf_counter()

        try:
            result = await node(state)

        except Exception:
            # -----------------------------------------------
            # Record timing even when the wrapped node fails.
            # -----------------------------------------------

            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            timings = _get_timings(
                state
            )

            timings["stages"][
                stage_name
            ] = round(
                duration_ms,
                3,
            )

            # Timing must never mask the original exception.
            state["timings"] = timings

            raise

        # ----------------------------------------------------
        # SUCCESSFUL NODE EXECUTION
        # ----------------------------------------------------

        duration_ms = (
            perf_counter() - started_at
        ) * 1000

        timings = _get_timings(
            result
        )

        timings["stages"][
            stage_name
        ] = round(
            duration_ms,
            3,
        )

        return {
            **result,
            "timings": timings,
        }

    return timed_node


def record_total_query_time(
    state: QueryState,
    duration_ms: float,
) -> QueryState:
    """
    Record total end-to-end workflow execution time.

    The result remains fully JSON-serializable.
    """

    timings = _get_timings(
        state
    )

    timings["total_ms"] = round(
        duration_ms,
        3,
    )

    return {
        **state,
        "timings": timings,
    }