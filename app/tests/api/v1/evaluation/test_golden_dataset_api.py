from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


TRACE_ID = "8416eeadc888ead049d6734e4dd6b62d"


class MockLangfuseGoldenDatasetService:
    def __init__(self) -> None:
        self.received_trace_ids: list[str] = []

    def build_from_traces(
        self,
        trace_ids: list[str],
    ) -> list[dict]:
        self.received_trace_ids = trace_ids

        return [
            {
                "id": "golden-candidate-1",
                "trace_id": trace_id,
                "question": "tell me Business Structure ?",
                "tenant_id": "contextops",
                "observed_answer": (
                    "Common business structures and example "
                    "supporting evidence include..."
                ),
                "expected_answer": None,
                "expected_source_ids": [],
                "review": {
                    "status": "pending",
                },
            }
            for trace_id in trace_ids
        ]


def test_golden_dataset_endpoint_exists() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/openapi.json",
        )

    assert response.status_code == 200

    openapi = response.json()

    assert (
        "/api/v1/evaluation/golden-dataset"
        in openapi["paths"]
    )

    assert (
        "post"
        in openapi["paths"][
            "/api/v1/evaluation/golden-dataset"
        ]
    )


def test_generate_golden_dataset() -> None:
    mock_service = MockLangfuseGoldenDatasetService()

    with patch(
        "app.api.v1.evaluation.router.LangfuseGoldenDatasetService",
        return_value=mock_service,
    ):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/evaluation/golden-dataset",
                json={
                    "trace_ids": [
                        TRACE_ID,
                    ],
                },
            )

    assert response.status_code == 200

    body = response.json()

    assert "examples" in body
    assert len(body["examples"]) == 1

    example = body["examples"][0]

    assert example["trace_id"] == TRACE_ID

    assert (
        example["question"]
        == "tell me Business Structure ?"
    )

    assert (
        example["tenant_id"]
        == "contextops"
    )

    assert (
        example["expected_answer"]
        is None
    )

    assert (
        example["expected_source_ids"]
        == []
    )

    assert (
        example["review"]["status"]
        == "pending"
    )

    assert (
        mock_service.received_trace_ids
        == [TRACE_ID]
    )


def test_generate_golden_dataset_requires_trace_ids() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluation/golden-dataset",
            json={
                "trace_ids": [],
            },
        )

    assert response.status_code == 422
