import json
import os
from pathlib import Path
from typing import Any

import pytest

from app.observability.langfuse import (
    configure_langfuse,
    get_langfuse_client,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TRACE_ID = (
    "8416eeadc888ead049d6734e4dd6b62d"
)

TARGET_OBSERVATIONS = [
    "hybrid_retrieval",
    "rerank",
    "retrieval_validation",
    "route_after_retrieval_validation",
    "llm_generation",
    "grounding",
    "response",
]

OUTPUT_DIRECTORY = Path(
    "app/tests/evaluation/artifacts"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "langfuse_trace_inspection.json"
)


# ============================================================
# TRACE ID
# ============================================================


def _get_trace_id() -> str:
    """
    Resolve the Langfuse trace ID.

    The environment variable allows the same test to be reused
    with another trace without changing this file.

    Example:

        $env:CONTEXTOPS_LANGFUSE_TRACE_ID="trace-id"
    """

    trace_id = os.getenv(
        "CONTEXTOPS_LANGFUSE_TRACE_ID",
        DEFAULT_TRACE_ID,
    )

    trace_id = trace_id.strip()

    if not trace_id:
        raise ValueError(
            "CONTEXTOPS_LANGFUSE_TRACE_ID is empty."
        )

    return trace_id


# ============================================================
# JSON SERIALIZATION
# ============================================================


def _to_jsonable(value: Any) -> Any:
    """
    Convert Langfuse SDK response objects into JSON-safe data.

    The structure is intentionally preserved as much as
    possible. We do not transform the observation payload into
    the future golden-dataset schema yet.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _to_jsonable(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(
                value.model_dump(
                    mode="json"
                )
            )
        except TypeError:
            return _to_jsonable(
                value.model_dump()
            )

    if hasattr(value, "dict"):
        try:
            return _to_jsonable(
                value.dict()
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return _to_jsonable(
            vars(value)
        )

    return str(value)


# ============================================================
# OBSERVATION HELPERS
# ============================================================


def _get_observation_name(
    observation: Any,
) -> str | None:
    """
    Extract the observation name from either a dictionary or
    Langfuse SDK object.
    """

    if isinstance(observation, dict):
        name = observation.get("name")

        if isinstance(name, str):
            return name

        return None

    name = getattr(
        observation,
        "name",
        None,
    )

    if isinstance(name, str):
        return name

    return None


def _extract_observations(
    response: Any,
) -> list[Any]:
    """
    Extract observation rows from the Langfuse API response.

    Langfuse API responses expose observation rows through
    response.data.
    """

    if response is None:
        return []

    if isinstance(response, dict):
        data = response.get("data")

        if isinstance(data, list):
            return data

        if isinstance(data, tuple):
            return list(data)

        return []

    data = getattr(
        response,
        "data",
        None,
    )

    if isinstance(data, list):
        return data

    if isinstance(data, tuple):
        return list(data)

    return []


# ============================================================
# LANGFUSE API
# ============================================================


def _create_langfuse_client() -> Any:
    """
    Configure Langfuse and create the client.

    This is important because the ContextOps application
    configures LANGFUSE_* environment variables inside
    configure_langfuse().
    """

    configured = configure_langfuse()

    if not configured:
        pytest.fail(
            "Langfuse tracing is not configured. "
            "Check LANGFUSE_TRACING, "
            "LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY."
        )

    client = get_langfuse_client()

    if client is None:
        pytest.fail(
            "Langfuse client is not available."
        )

    return client


def _fetch_observations(
    client: Any,
    trace_id: str,
) -> Any:
    """
    Fetch all observations belonging to the supplied trace.

    Langfuse Python SDK v4 exposes the Observations API through:

        client.api.observations.get_many(...)

    We request all fields needed for the later golden-dataset
    design instead of restricting the response to only core
    fields.
    """

    api = getattr(
        client,
        "api",
        None,
    )

    if api is None:
        raise RuntimeError(
            "Langfuse client does not expose client.api."
        )

    observations_api = getattr(
        api,
        "observations",
        None,
    )

    if observations_api is None:
        raise RuntimeError(
            "Langfuse client does not expose "
            "client.api.observations."
        )

    get_many = getattr(
        observations_api,
        "get_many",
        None,
    )

    if get_many is None:
        raise RuntimeError(
            "Langfuse observations API does not expose "
            "get_many()."
        )

    return get_many(
        trace_id=trace_id,
        fields=(
            "core,"
            "basic,"
            "time,"
            "io,"
            "metadata,"
            "model,"
            "usage,"
            "prompt,"
            "trace_context"
        ),
        limit=1000,
    )


# ============================================================
# TEST
# ============================================================


def test_inspect_langfuse_trace_observations() -> None:
    """
    Inspect an existing ContextOps Langfuse trace.

    This test does not execute /query.

    It reads an already-created Langfuse trace and inspects the
    actual structure of:

        1. hybrid_retrieval
        2. rerank
        3. retrieval_validation
        4. route_after_retrieval_validation

    The output is saved without converting it into the final
    golden-dataset schema.

    The schema will be designed after seeing the real Langfuse
    observation payloads.
    """

    trace_id = _get_trace_id()

    # --------------------------------------------------------
    # Configure Langfuse before creating the client
    # --------------------------------------------------------

    client = _create_langfuse_client()

    # --------------------------------------------------------
    # Fetch observations
    # --------------------------------------------------------

    response = _fetch_observations(
        client=client,
        trace_id=trace_id,
    )

    observations = _extract_observations(
        response
    )

    if not observations:
        pytest.skip(
            "No Langfuse observations were returned for "
            f"trace_id={trace_id}"
        )

    # --------------------------------------------------------
    # Convert complete response to JSON
    # --------------------------------------------------------

    raw_response = _to_jsonable(
        response
    )

    # --------------------------------------------------------
    # Find target observations
    # --------------------------------------------------------

    matched_observations: dict[
        str,
        list[Any],
    ] = {
        name: []
        for name in TARGET_OBSERVATIONS
    }

    for observation in observations:
        name = _get_observation_name(
            observation
        )

        if name in matched_observations:
            matched_observations[
                name
            ].append(
                _to_jsonable(
                    observation
                )
            )

    # --------------------------------------------------------
    # Build inspection artifact
    # --------------------------------------------------------

    inspection: dict[str, Any] = {
        "trace_id": trace_id,
        "total_observations": len(
            observations
        ),
        "target_observations": (
            TARGET_OBSERVATIONS
        ),
        "matched_observation_counts": {
            name: len(items)
            for name, items in (
                matched_observations.items()
            )
        },
        "matched_observations": (
            matched_observations
        ),
        "raw_response": raw_response,
    }

    # --------------------------------------------------------
    # Save inspection artifact
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            inspection,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console header
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "LANGFUSE TRACE INSPECTION"
    )
    print("=" * 80)

    print(
        f"Trace ID: {trace_id}"
    )

    print(
        f"Total observations: "
        f"{len(observations)}"
    )

    print(
        f"Inspection file: "
        f"{OUTPUT_FILE}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Print all observation names
    # --------------------------------------------------------

    print()
    print(
        "ALL OBSERVATION NAMES"
    )
    print("-" * 80)

    all_names: list[str] = []

    for observation in observations:
        name = _get_observation_name(
            observation
        )

        if name is not None:
            all_names.append(name)

    for index, name in enumerate(
        all_names,
        start=1,
    ):
        print(
            f"{index:03d}. {name}"
        )

    # --------------------------------------------------------
    # Print target observations
    # --------------------------------------------------------

    for observation_name in (
        TARGET_OBSERVATIONS
    ):
        items = matched_observations[
            observation_name
        ]

        print()
        print("=" * 80)

        print(
            f"TARGET OBSERVATION: "
            f"{observation_name}"
        )

        print("=" * 80)

        print(
            f"Matches: {len(items)}"
        )

        if not items:
            print(
                "NOT FOUND"
            )
            continue

        for index, observation in enumerate(
            items,
            start=1,
        ):
            print()
            print(
                f"--- MATCH {index} ---"
            )

            if isinstance(
                observation,
                dict,
            ):
                print(
                    "Observation keys:"
                )

                for key in observation:
                    print(
                        f"  - {key}"
                    )

                print()
                print(
                    "Observation JSON:"
                )

                print(
                    json.dumps(
                        observation,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            else:
                print(
                    json.dumps(
                        _to_jsonable(
                            observation
                        ),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TARGET OBSERVATION SUMMARY"
    )
    print("=" * 80)

    for observation_name in (
        TARGET_OBSERVATIONS
    ):
        count = len(
            matched_observations[
                observation_name
            ]
        )

        status = (
            "FOUND"
            if count > 0
            else "NOT FOUND"
        )

        print(
            f"{observation_name}: "
            f"{status} ({count})"
        )

    print()
    print(
        "Raw inspection saved to:"
    )

    print(
        str(OUTPUT_FILE)
    )

    print()

    # --------------------------------------------------------
    # The trace itself must exist.
    #
    # Individual target observations are intentionally NOT
    # asserted here because we first want to inspect the actual
    # production trace structure.
    # --------------------------------------------------------

    assert len(observations) > 0
