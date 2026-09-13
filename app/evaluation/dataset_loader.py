import json
from pathlib import Path
from typing import Any

from app.evaluation.models import EvaluationExample


class EvaluationDatasetLoader:
    """
    Load and validate ContextOps golden evaluation datasets.

    Expected dataset format:

        [
            {
                "question": "...",
                "tenant_id": "...",
                "expected_answer": "...",
                "expected_source_ids": [],
                "conversation_id": null,
                "filters": null
            }
        ]
    """

    def load(
        self,
        path: str | Path,
    ) -> list[EvaluationExample]:
        """
        Load evaluation examples from a JSON file.
        """

        dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: "
                f"{dataset_path}"
            )

        if not dataset_path.is_file():
            raise ValueError(
                f"Evaluation dataset path is not a file: "
                f"{dataset_path}"
            )

        try:
            raw_content = dataset_path.read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            raise OSError(
                f"Failed to read evaluation dataset: "
                f"{dataset_path}"
            ) from exc

        try:
            payload = json.loads(
                raw_content
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Evaluation dataset contains invalid "
                f"JSON: {dataset_path}"
            ) from exc

        if not isinstance(
            payload,
            list,
        ):
            raise ValueError(
                "Evaluation dataset root must be a JSON array."
            )

        examples: list[EvaluationExample] = []

        for index, item in enumerate(payload):
            examples.append(
                self._parse_example(
                    item,
                    index=index,
                )
            )

        return examples

    # =========================================================
    # EXAMPLE PARSING
    # =========================================================

    def _parse_example(
        self,
        item: Any,
        *,
        index: int,
    ) -> EvaluationExample:
        """
        Validate and convert one JSON object into an
        EvaluationExample.
        """

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Evaluation example at index {index} "
                f"must be a JSON object."
            )

        question = self._required_string(
            item,
            "question",
            index=index,
        )

        tenant_id = self._required_string(
            item,
            "tenant_id",
            index=index,
        )

        expected_answer = self._optional_string(
            item,
            "expected_answer",
            index=index,
        )

        expected_source_ids = (
            self._source_ids(
                item,
                index=index,
            )
        )

        conversation_id = self._optional_string(
            item,
            "conversation_id",
            index=index,
        )

        filters = self._filters(
            item,
            index=index,
        )

        return EvaluationExample(
            question=question,
            tenant_id=tenant_id,
            expected_answer=expected_answer,
            expected_source_ids=(
                expected_source_ids
            ),
            conversation_id=conversation_id,
            filters=filters,
        )

    # =========================================================
    # STRING VALIDATION
    # =========================================================

    @staticmethod
    def _required_string(
        item: dict[str, Any],
        field_name: str,
        *,
        index: int,
    ) -> str:
        """
        Read a required non-empty string.
        """

        value = item.get(
            field_name
        )

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                f"Evaluation example {index}: "
                f"'{field_name}' must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"Evaluation example {index}: "
                f"'{field_name}' cannot be empty."
            )

        return value

    @staticmethod
    def _optional_string(
        item: dict[str, Any],
        field_name: str,
        *,
        index: int,
    ) -> str | None:
        """
        Read an optional string.
        """

        value = item.get(
            field_name
        )

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                f"Evaluation example {index}: "
                f"'{field_name}' must be a string or null."
            )

        value = value.strip()

        return value or None

    # =========================================================
    # SOURCE IDS
    # =========================================================

    @staticmethod
    def _source_ids(
        item: dict[str, Any],
        *,
        index: int,
    ) -> list[str]:
        """
        Validate expected source identifiers.
        """

        value = item.get(
            "expected_source_ids",
            [],
        )

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            raise ValueError(
                f"Evaluation example {index}: "
                f"'expected_source_ids' must be an array."
            )

        source_ids: list[str] = []

        for source_index, source_id in enumerate(
            value
        ):
            if not isinstance(
                source_id,
                str,
            ):
                raise ValueError(
                    f"Evaluation example {index}: "
                    f"'expected_source_ids[{source_index}]' "
                    f"must be a string."
                )

            normalized_source_id = (
                source_id.strip()
            )

            if not normalized_source_id:
                raise ValueError(
                    f"Evaluation example {index}: "
                    f"'expected_source_ids[{source_index}]' "
                    f"cannot be empty."
                )

            source_ids.append(
                normalized_source_id
            )

        return source_ids

    # =========================================================
    # FILTERS
    # =========================================================

    @staticmethod
    def _filters(
        item: dict[str, Any],
        *,
        index: int,
    ) -> dict[str, Any] | None:
        """
        Validate optional retrieval filters.
        """

        value = item.get(
            "filters"
        )

        if value is None:
            return None

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                f"Evaluation example {index}: "
                f"'filters' must be an object or null."
            )

        return dict(value)