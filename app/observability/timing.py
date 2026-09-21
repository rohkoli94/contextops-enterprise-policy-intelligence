from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any
import logging

from app.rag.workflow.state import QueryState


logger = logging.getLogger("contextops.graph")


def _get_timings(
    state: QueryState,
) -> dict[str, Any]:
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
    Wrap a workflow node with timing and
    structured lifecycle logging.

    The wrapper:

    - validates the node name
    - logs node start
    - measures execution duration
    - records successful node duration
    - records failed node duration
    - logs node completion
    - logs node failure
    - preserves the original exception
    """

    if not name or not name.strip():
        raise ValueError(
            "Timing node name cannot be empty"
        )

    async def timed_node(
        state: QueryState,
    ) -> QueryState:

        if name == "llm_generation":
            logger.info(
                "\n"
                "============================================================\n"
                "🤖 NODE STARTED: llm_generation\n"
                "============================================================"
            )
        else:
            logger.info(
                "▶ NODE STARTED: %s",
                name,
            )

        started_at = perf_counter()

        try:
            result = await node(state)

            elapsed_ms = (
                perf_counter() - started_at
            ) * 1000

            timings = _get_timings(
                state
            )

            timings["stages"][name] = round(
                elapsed_ms,
                3,
            )

            if name == "llm_generation":
                logger.info(
                    "\n"
                    "============================================================\n"
                    "🤖 NODE COMPLETED: llm_generation\n"
                    "⏱ duration_ms=%.3f\n"
                    "============================================================",
                    elapsed_ms,
                )
            else:
                logger.info(
                    "✓ NODE COMPLETED: %s | "
                    "duration_ms=%.3f",
                    name,
                    elapsed_ms,
                )

            return {
                **result,
                "timings": timings,
            }

        except Exception:
            elapsed_ms = (
                perf_counter() - started_at
            ) * 1000

            timings = _get_timings(
                state
            )

            timings["stages"][name] = round(
                elapsed_ms,
                3,
            )

            # Update the original state as well.
            # This is important because the node
            # raised an exception and therefore
            # cannot return a new state object.
            state["timings"] = timings

            if name == "llm_generation":
                logger.exception(
                    "\n"
                    "============================================================\n"
                    "🚨 NODE FAILED: llm_generation\n"
                    "⏱ duration_ms=%.3f\n"
                    "============================================================",
                    elapsed_ms,
                )
            else:
                logger.exception(
                    "✗ NODE FAILED: %s | "
                    "duration_ms=%.3f",
                    name,
                    elapsed_ms,
                )

            raise

    return timed_node


def record_total_query_time(
    state: QueryState,
    elapsed_ms: float,
) -> QueryState:
    """
    Record total workflow execution time.
    """

    timings = _get_timings(
        state
    )

    timings["total_ms"] = round(
        elapsed_ms,
        3,
    )

    return {
        **state,
        "timings": timings,
    }