from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.golden_dataset_builder import (
    TARGET_OBSERVATIONS,
)
from app.evaluation.langfuse_golden_dataset_service import (
    LangfuseGoldenDatasetService,
)


TRACE_ID = "8416eeadc888ead049d6734e4dd6b62d"

ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "langfuse_trace_inspection.json"
)


@dataclass
class MockObservation:
    payload: dict[str, Any]

    def model_dump(
        self,
        mode: str | None = None,
    ) -> dict[str, Any]:
        return self.payload


class MockObservationResponse:
    def __init__(
        self,
        observations: list[MockObservation],
    ) -> None:
        self.data = observations


class MockObservationsAPI:
    def __init__(
        self,
        observations: list[MockObservation],
    ) -> None:
        self.observations = observations
        self.calls: list[dict[str, Any]] = []

    def get_many(
        self,
        **kwargs: Any,
    ) -> MockObservationResponse:
        self.calls.append(kwargs)

        return MockObservationResponse(
            self.observations,
        )


class MockAPI:
    def __init__(
        self,
        observations: list[MockObservation],
    ) -> None:
        self.observations = MockObservationsAPI(
            observations,
        )


class MockLangfuseClient:
    def __init__(
        self,
        observations: list[MockObservation],
    ) -> None:
        self.api = MockAPI(observations)


def _load_inspection_artifact() -> dict[str, Any]:
    import json

    return json.loads(
        ARTIFACT_PATH.read_text(
            encoding="utf-8",
        )
    )


def _create_mock_observations() -> list[MockObservation]:
    artifact = _load_inspection_artifact()

    matched_observations = artifact[
        "matched_observations"
    ]

    observations: list[MockObservation] = []

    for matches in matched_observations.values():
        for payload in matches:
            observations.append(
                MockObservation(
                    payload=payload,
                )
            )

    return observations


def test_fetch_observations_uses_langfuse_api() -> None:
    observations = _create_mock_observations()

    client = MockLangfuseClient(
        observations=observations,
    )

    service = LangfuseGoldenDatasetService(
        client=client,
    )

    result = service.fetch_observations(
        trace_id=TRACE_ID,
    )

    assert len(result) == len(observations)

    call = client.api.observations.calls[0]

    assert call["trace_id"] == TRACE_ID
    assert call["limit"] == 1000

    assert (
        call["fields"]
        == (
            "core,"
            "basic,"
            "time,"
            "io,"
            "metadata,"
            "model,"
            "usage,"
            "prompt,"
            "trace_context"
        )
    )


def test_build_from_trace_creates_golden_candidate() -> None:
    observations = _create_mock_observations()

    client = MockLangfuseClient(
        observations=observations,
    )

    service = LangfuseGoldenDatasetService(
        client=client,
    )

    candidate = service.build_from_trace(
        trace_id=TRACE_ID,
    )

    assert candidate["trace_id"] == TRACE_ID

    assert (
        candidate["question"]
        == "tell me Business Structure ?"
    )

    assert candidate["tenant_id"] == "contextops"

    assert (
        candidate["conversation_id"]
        == "8a1bebec-ae0a-465e-8043-02d86e3e13a2"
    )

    assert candidate["observed_answer"]

    assert candidate["expected_answer"] is None

    assert candidate["expected_source_ids"] == []

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    assert (
        candidate["retrieval"]["retrieval_sufficient"]
        is True
    )

    assert (
        candidate["retrieval"]["retrieval_confidence"]
        == "sufficient"
    )

    assert (
        candidate["retrieval"]["retrieval_score"]
        == 2.297222375869751
    )

    assert (
        candidate["retrieval"]["retrieval_signals"][
            "document_count"
        ]
        == 20
    )

    assert (
        candidate["retrieval"]["retrieval_signals"][
            "usable_document_count"
        ]
        == 20
    )

    assert (
        candidate["retrieval"]["retrieval_signals"][
            "reranker_score_count"
        ]
        == 20
    )

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------

    assert (
        candidate["routing"]["retrieval_validation_route"]
        == "contextops"
    )

    # --------------------------------------------------------
    # Retrieved sources
    # --------------------------------------------------------

    assert candidate["retrieved_sources"]

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    assert candidate["citations"]

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    assert (
        candidate["grounding"]["status"]
        == "grounded"
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    assert (
        candidate["context"]["token_count"]
        == 1006
    )


def test_build_from_trace_requires_all_target_observations() -> None:
    observations = _create_mock_observations()

    filtered = [
        observation
        for observation in observations
        if observation.payload.get("name")
        != "response"
    ]

    client = MockLangfuseClient(
        observations=filtered,
    )

    service = LangfuseGoldenDatasetService(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="missing required observations",
    ):
        service.build_from_trace(
            trace_id=TRACE_ID,
        )


def test_required_observations_match_expected_trace_structure() -> None:
    observations = _create_mock_observations()

    names = {
        observation.payload.get("name")
        for observation in observations
    }

    for name in TARGET_OBSERVATIONS:
        assert name in names
