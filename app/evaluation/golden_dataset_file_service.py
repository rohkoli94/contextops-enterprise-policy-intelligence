from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.golden_source_identifier import (
    GoldenSourceIdentifier,
)
from app.evaluation.langfuse_golden_dataset_service import (
    LangfuseGoldenDatasetService,
)


BASE_DIR = Path(__file__).resolve().parent

GOLDEN_DATASET_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_golden.json"
)

GOLDEN_SOURCE_CANDIDATES_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_golden_source_candidates.json"
)


class GoldenDatasetFileService:
    """
    Generate and overwrite both ContextOps evaluation dataset files.

    Each Langfuse trace is processed independently.

    Valid retrieval traces are converted into golden-dataset
    candidates.

    Traces that do not contain the required retrieval observations
    are skipped and reported instead of failing the entire operation.
    """

    def __init__(
        self,
        langfuse_service: (
            LangfuseGoldenDatasetService | None
        ) = None,
        source_identifier: (
            GoldenSourceIdentifier | None
        ) = None,
    ) -> None:
        self.langfuse_service = (
            langfuse_service
            or LangfuseGoldenDatasetService()
        )

        self.source_identifier = (
            source_identifier
            or GoldenSourceIdentifier()
        )

    @staticmethod
    def _write_json(
        path: Path,
        data: Any,
    ) -> None:
        """
        Write JSON and overwrite an existing file.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _build_examples(
        self,
        trace_ids: list[str],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, str]],
    ]:
        """
        Build golden examples independently for every trace.

        A single invalid/incomplete trace must not prevent valid
        traces from being processed.

        Returns:
            (
                valid_golden_examples,
                skipped_traces,
            )
        """

        golden_examples: list[
            dict[str, Any]
        ] = []

        skipped_traces: list[
            dict[str, str]
        ] = []

        for trace_id in trace_ids:

            normalized_trace_id = (
                trace_id.strip()
            )

            if not normalized_trace_id:

                continue

            try:

                example = (
                    self.langfuse_service.build_from_trace(
                        trace_id=normalized_trace_id,
                    )
                )

                golden_examples.append(
                    example
                )

            except Exception as exc:

                skipped_traces.append(
                    {
                        "trace_id": normalized_trace_id,
                        "reason": str(exc),
                    }
                )

        return (
            golden_examples,
            skipped_traces,
        )

    async def generate(
        self,
        trace_ids: list[str],
    ) -> dict[str, Any]:
        """
        Generate both evaluation files.

        Processing behavior:

        1. Validate that at least one trace ID was supplied.
        2. Process every trace independently.
        3. Keep valid retrieval-backed traces.
        4. Skip traces missing required observations.
        5. Fail only when no valid traces remain.
        6. Overwrite contextops_golden.json.
        7. Generate source candidates from the valid examples.
        8. Overwrite contextops_golden_source_candidates.json.
        9. Return generated data plus skipped-trace information.
        """

        if not trace_ids:

            raise ValueError(
                "At least one Langfuse trace ID is required."
            )

        # --------------------------------------------------------
        # Normalize and de-duplicate trace IDs while preserving
        # the original order supplied by the user.
        # --------------------------------------------------------

        normalized_trace_ids: list[str] = []

        seen_trace_ids: set[str] = set()

        for trace_id in trace_ids:

            normalized_trace_id = (
                trace_id.strip()
            )

            if not normalized_trace_id:
                continue

            if normalized_trace_id in seen_trace_ids:
                continue

            seen_trace_ids.add(
                normalized_trace_id
            )

            normalized_trace_ids.append(
                normalized_trace_id
            )

        if not normalized_trace_ids:

            raise ValueError(
                "At least one non-empty Langfuse trace ID is required."
            )

        # --------------------------------------------------------
        # Process each trace independently.
        # --------------------------------------------------------

        (
            golden_examples,
            skipped_traces,
        ) = self._build_examples(
            trace_ids=normalized_trace_ids,
        )

        # --------------------------------------------------------
        # Do not overwrite the existing files with empty data.
        # --------------------------------------------------------

        if not golden_examples:

            reasons = "; ".join(
                (
                    f"{item['trace_id']}: "
                    f"{item['reason']}"
                )
                for item in skipped_traces
            )

            raise ValueError(
                (
                    "No valid Langfuse traces were found. "
                    f"All supplied traces were skipped. "
                    f"Details: {reasons}"
                )
            )

        # --------------------------------------------------------
        # Overwrite contextops_golden.json
        # --------------------------------------------------------

        self._write_json(
            GOLDEN_DATASET_FILE,
            golden_examples,
        )

        # --------------------------------------------------------
        # Generate source candidates only from valid golden
        # examples.
        # --------------------------------------------------------

        source_candidates = (
            await self.source_identifier.generate(
                examples=golden_examples,
            )
        )

        # --------------------------------------------------------
        # Overwrite contextops_golden_source_candidates.json
        # --------------------------------------------------------

        self._write_json(
            GOLDEN_SOURCE_CANDIDATES_FILE,
            source_candidates,
        )

        # --------------------------------------------------------
        # Return generated data and processing information.
        # --------------------------------------------------------

        return {
            "golden_dataset_file": str(
                GOLDEN_DATASET_FILE
            ),
            "golden_source_candidates_file": str(
                GOLDEN_SOURCE_CANDIDATES_FILE
            ),
            "example_count": len(
                golden_examples
            ),
            "candidate_count": len(
                source_candidates
            ),
            "golden_dataset": golden_examples,
            "source_candidates": source_candidates,
            "processed_trace_ids": [
                example.get(
                    "trace_id"
                )
                for example in golden_examples
            ],
            "skipped_traces": skipped_traces,
            "skipped_count": len(
                skipped_traces
            ),
        }