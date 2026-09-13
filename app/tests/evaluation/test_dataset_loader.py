import json

import pytest

from app.evaluation.dataset_loader import (
    EvaluationDatasetLoader,
)


def write_dataset(
    tmp_path,
    payload,
):
    path = (
        tmp_path
        / "evaluation.json"
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return path


def test_loader_reads_valid_dataset(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "What is the leave policy?",
                "tenant_id": "tenant-001",
                "expected_answer": (
                    "Employees receive annual leave."
                ),
                "expected_source_ids": [
                    "chunk-001"
                ],
                "conversation_id": None,
                "filters": {
                    "content_type": "text",
                },
            }
        ],
    )

    examples = (
        EvaluationDatasetLoader().load(
            path
        )
    )

    assert len(examples) == 1

    example = examples[0]

    assert example.question == (
        "What is the leave policy?"
    )

    assert example.tenant_id == (
        "tenant-001"
    )

    assert example.expected_answer == (
        "Employees receive annual leave."
    )

    assert example.expected_source_ids == [
        "chunk-001"
    ]

    assert example.filters == {
        "content_type": "text",
    }


def test_loader_rejects_missing_file(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "missing.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation dataset not found",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_rejects_invalid_json(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "invalid.json"
    )

    path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_rejects_non_array_root(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        {
            "question": "test"
        },
    )

    with pytest.raises(
        ValueError,
        match="root must be a JSON array",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_rejects_empty_question(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "   ",
                "tenant_id": "tenant-001",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="question.*cannot be empty",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_rejects_missing_tenant(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "What is the policy?",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="tenant_id.*must be a string",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_rejects_invalid_source_ids(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "What is the policy?",
                "tenant_id": "tenant-001",
                "expected_source_ids": "chunk-001",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="expected_source_ids.*array",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_rejects_invalid_filters(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "What is the policy?",
                "tenant_id": "tenant-001",
                "filters": [],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="filters.*object",
    ):
        EvaluationDatasetLoader().load(
            path
        )


def test_loader_allows_optional_values_to_be_null(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "What is the policy?",
                "tenant_id": "tenant-001",
                "expected_answer": None,
                "expected_source_ids": None,
                "conversation_id": None,
                "filters": None,
            }
        ],
    )

    examples = (
        EvaluationDatasetLoader().load(
            path
        )
    )

    example = examples[0]

    assert example.expected_answer is None
    assert example.expected_source_ids == []
    assert example.conversation_id is None
    assert example.filters is None