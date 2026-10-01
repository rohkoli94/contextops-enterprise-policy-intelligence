import json
from pathlib import Path

from app.evaluation.golden_dataset_builder import (
    TARGET_OBSERVATIONS,
    build_golden_candidate,
)


ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "langfuse_trace_inspection.json"
)


def _load_observations() -> tuple[
    str,
    list[dict],
]:
    payload = json.loads(
        ARTIFACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    trace_id = payload["trace_id"]

    matched = payload[
        "matched_observations"
    ]

    observations = []

    for matches in matched.values():
        observations.extend(matches)

    return trace_id, observations


def test_production_builder_contains_required_observations():
    trace_id, observations = (
        _load_observations()
    )

    names = {
        observation.get("name")
        for observation in observations
    }

    assert TARGET_OBSERVATIONS <= names


def test_production_builder_creates_golden_candidate():
    trace_id, observations = (
        _load_observations()
    )

    candidate = build_golden_candidate(
        trace_id=trace_id,
        observations=observations,
    )

    assert candidate["trace_id"] == trace_id

    assert candidate["question"] == (
        "tell me Business Structure ?"
    )

    assert candidate["tenant_id"] == (
        "contextops"
    )

    assert candidate["conversation_id"] == (
        "8a1bebec-ae0a-465e-8043-02d86e3e13a2"
    )

    assert candidate["observed_answer"]

    assert "Sole Proprietorship" in (
        candidate["observed_answer"]
    )

    assert "Private Limited Company" in (
        candidate["observed_answer"]
    )

    assert candidate["expected_answer"] is None

    assert candidate[
        "expected_source_ids"
    ] == []

    assert candidate["retrieval"][
        "retrieval_sufficient"
    ] is True

    assert candidate["retrieval"][
        "retrieval_confidence"
    ] == "sufficient"

    assert candidate["routing"][
        "retrieval_validation_route"
    ] == "contextops"

    assert candidate[
        "retrieved_sources"
    ]

    assert candidate[
        "citations"
    ]

    assert candidate[
        "grounding"
    ]["status"] == "grounded"

    assert candidate[
        "review"
    ]["status"] == "pending"
