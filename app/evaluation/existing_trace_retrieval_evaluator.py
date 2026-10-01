from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.langfuse_golden_dataset_service import (
    LangfuseGoldenDatasetService,
)
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    calculate_retrieval_metrics,
)
from app.observability.langfuse import (
    configure_langfuse,
)


BASE_DIR = Path(__file__).resolve().parent


GOLDEN_DATASET_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_golden.json"
)


TRACE_SOURCE_REVIEW_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_trace_source_review.json"
)


OUTPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_retrieval_evaluation.json"
)


class ExistingTraceRetrievalEvaluator:
    """
    Evaluate retrieval using already-existing Langfuse traces.

    IMPORTANT:

    This evaluator does NOT:

        - call QueryService
        - execute /query
        - perform new vector retrieval
        - call Qdrant
        - call an LLM

    It only reads existing Langfuse observations and compares:

        expected_source_ids
                    vs
        actual retrieved_sources

    Expected source IDs are resolved in this order:

        1. Explicit reviewed trace mapping
        2. Golden dataset question mapping

    The explicit trace mapping is important because an existing
    production-like trace does not necessarily correspond to one
    of the 12 current golden-dataset questions.
    """

    def __init__(
        self,
        *,
        langfuse_service: (
            LangfuseGoldenDatasetService | None
        ) = None,
        golden_dataset_file: Path | None = None,
        trace_source_review_file: Path | None = None,
        output_file: Path | None = None,
    ) -> None:
        # --------------------------------------------------------
        # Configure Langfuse BEFORE creating the client/service.
        # --------------------------------------------------------

        configured = configure_langfuse()

        if not configured:
            raise RuntimeError(
                "Langfuse is not configured. "
                "Check LANGFUSE_TRACING, "
                "LANGFUSE_PUBLIC_KEY, "
                "LANGFUSE_SECRET_KEY and "
                "LANGFUSE_BASE_URL."
            )

        self.langfuse_service = (
            langfuse_service
            or LangfuseGoldenDatasetService()
        )

        self.golden_dataset_file = (
            golden_dataset_file
            or GOLDEN_DATASET_FILE
        )

        self.trace_source_review_file = (
            trace_source_review_file
            or TRACE_SOURCE_REVIEW_FILE
        )

        self.output_file = (
            output_file
            or OUTPUT_FILE
        )

    # =========================================================
    # GOLDEN DATASET
    # =========================================================

    def load_golden_dataset(
        self,
    ) -> list[dict[str, Any]]:
        """
        Load the reviewed ContextOps golden dataset.
        """

        if not self.golden_dataset_file.exists():
            raise FileNotFoundError(
                "Golden dataset not found: "
                f"{self.golden_dataset_file}"
            )

        payload = json.loads(
            self.golden_dataset_file.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, list):
            raise ValueError(
                "Golden dataset must contain a JSON array."
            )

        return payload

    def _golden_by_question(
        self,
        golden_dataset: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Index golden examples by normalized question.
        """

        indexed: dict[
            str,
            dict[str, Any],
        ] = {}

        for example in golden_dataset:
            question = example.get(
                "question"
            )

            if not isinstance(
                question,
                str,
            ):
                continue

            normalized = (
                question.strip().lower()
            )

            if normalized:
                indexed[normalized] = example

        return indexed

    # =========================================================
    # TRACE SOURCE REVIEW
    # =========================================================

    def load_trace_source_review(
        self,
    ) -> list[dict[str, Any]]:
        """
        Load explicit reviewed expected-source mappings for
        existing Langfuse traces.

        This file represents human-reviewed evidence mapping.

        It is intentionally separate from the 12-question
        golden dataset because existing production-like traces
        may contain questions that are not currently part of
        the golden dataset.
        """

        if not self.trace_source_review_file.exists():
            raise FileNotFoundError(
                "Trace source review file not found: "
                f"{self.trace_source_review_file}"
            )

        payload = json.loads(
            self.trace_source_review_file.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            payload,
            list,
        ):
            raise ValueError(
                "Trace source review must contain "
                "a JSON array."
            )

        return payload

    def _trace_review_by_id(
        self,
        reviews: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Index reviewed trace mappings by trace ID.
        """

        indexed: dict[
            str,
            dict[str, Any],
        ] = {}

        for review in reviews:
            trace_id = review.get(
                "trace_id"
            )

            if not isinstance(
                trace_id,
                str,
            ):
                continue

            trace_id = trace_id.strip()

            if not trace_id:
                continue

            indexed[trace_id] = review

        return indexed

    # =========================================================
    # EXPECTED SOURCE RESOLUTION
    # =========================================================

    def _resolve_expected_source_ids(
        self,
        *,
        trace_id: str,
        question: str,
        trace_reviews_by_id: dict[
            str,
            dict[str, Any],
        ],
        golden_by_question: dict[
            str,
            dict[str, Any],
        ],
    ) -> tuple[
        list[str],
        str,
    ]:
        """
        Resolve expected source IDs.

        Priority:

            1. Explicit trace review
            2. Golden dataset question mapping
        """

        # -----------------------------------------------------
        # Priority 1:
        # Explicit trace-level reviewed mapping.
        # -----------------------------------------------------

        trace_review = trace_reviews_by_id.get(
            trace_id
        )

        if trace_review is not None:
            expected_source_ids = (
                trace_review.get(
                    "expected_source_ids",
                    [],
                )
            )

            if not isinstance(
                expected_source_ids,
                list,
            ):
                raise ValueError(
                    "Trace review expected_source_ids "
                    "must be a list for trace: "
                    f"{trace_id}"
                )

            normalized_ids = [
                str(source_id)
                for source_id in expected_source_ids
                if source_id is not None
                and str(source_id).strip()
            ]

            if normalized_ids:
                return (
                    normalized_ids,
                    "trace_review",
                )

        # -----------------------------------------------------
        # Priority 2:
        # Golden dataset question mapping.
        # -----------------------------------------------------

        normalized_question = (
            question.strip().lower()
        )

        golden_example = (
            golden_by_question.get(
                normalized_question
            )
        )

        if golden_example is not None:
            expected_source_ids = (
                golden_example.get(
                    "expected_source_ids",
                    [],
                )
            )

            if not isinstance(
                expected_source_ids,
                list,
            ):
                raise ValueError(
                    "Golden example expected_source_ids "
                    "must be a list for question: "
                    f"{question!r}"
                )

            normalized_ids = [
                str(source_id)
                for source_id in expected_source_ids
                if source_id is not None
                and str(source_id).strip()
            ]

            if normalized_ids:
                return (
                    normalized_ids,
                    "golden_dataset",
                )

        raise ValueError(
            "No approved expected_source_ids found "
            f"for trace_id={trace_id}, "
            f"question={question!r}"
        )

    # =========================================================
    # RETRIEVED SOURCE EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_retrieved_source_ids(
        candidate: dict[str, Any],
    ) -> list[str]:
        """
        Extract chunk IDs from the existing trace's
        retrieved_sources.

        The order returned by Langfuse is preserved because
        retrieval metrics depend on rank.
        """

        retrieved_sources = candidate.get(
            "retrieved_sources",
            [],
        )

        if not isinstance(
            retrieved_sources,
            list,
        ):
            raise ValueError(
                "Trace retrieved_sources must be a list."
            )

        retrieved_source_ids: list[str] = []

        for source in retrieved_sources:
            if not isinstance(
                source,
                dict,
            ):
                continue

            chunk_id = source.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            chunk_id = str(
                chunk_id
            ).strip()

            if chunk_id:
                retrieved_source_ids.append(
                    chunk_id
                )

        return retrieved_source_ids

    # =========================================================
    # TRACE PROCESSING
    # =========================================================

    def evaluate_trace(
        self,
        *,
        trace_id: str,
        golden_by_question: dict[
            str,
            dict[str, Any],
        ],
        trace_reviews_by_id: dict[
            str,
            dict[str, Any],
        ],
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    ) -> dict[str, Any]:
        """
        Evaluate one existing Langfuse trace.
        """

        trace_id = trace_id.strip()

        if not trace_id:
            raise ValueError(
                "trace_id cannot be empty."
            )

        # -----------------------------------------------------
        # Fetch the existing trace.
        #
        # No new query is executed here.
        # -----------------------------------------------------

        candidate = (
            self.langfuse_service.build_from_trace(
                trace_id=trace_id
            )
        )

        question = candidate.get(
            "question"
        )

        if not isinstance(
            question,
            str,
        ) or not question.strip():
            raise ValueError(
                f"Trace {trace_id} does not contain "
                "a valid question."
            )

        # -----------------------------------------------------
        # Resolve human-reviewed expected source IDs.
        # -----------------------------------------------------

        (
            expected_source_ids,
            expected_source_origin,
        ) = self._resolve_expected_source_ids(
            trace_id=trace_id,
            question=question,
            trace_reviews_by_id=(
                trace_reviews_by_id
            ),
            golden_by_question=(
                golden_by_question
            ),
        )

        # -----------------------------------------------------
        # Extract the sources that were ACTUALLY retrieved
        # in the existing trace.
        # -----------------------------------------------------

        retrieved_source_ids = (
            self._extract_retrieved_source_ids(
                candidate
            )
        )

        # -----------------------------------------------------
        # Deterministic retrieval metrics.
        # -----------------------------------------------------

        metrics = calculate_retrieval_metrics(
            expected_source_ids=(
                expected_source_ids
            ),
            retrieved_source_ids=(
                retrieved_source_ids
            ),
            k_values=k_values,
        )

        return {
            "trace_id": trace_id,
            "question": question,
            "tenant_id": candidate.get(
                "tenant_id"
            ),
            "expected_source_origin": (
                expected_source_origin
            ),
            "expected_source_ids": list(
                metrics.expected_source_ids
            ),
            "retrieved_source_ids": list(
                metrics.retrieved_source_ids
            ),
            "recall_at_k": {
                str(k): value
                for k, value in (
                    metrics.recall_at_k.items()
                )
            },
            "precision_at_k": {
                str(k): value
                for k, value in (
                    metrics.precision_at_k.items()
                )
            },
            "mrr": metrics.reciprocal_rank,
            "first_relevant_rank": (
                metrics.first_relevant_rank
            ),
            "ndcg_at_k": {
                str(k): value
                for k, value in (
                    metrics.ndcg_at_k.items()
                )
            },
        }

    # =========================================================
    # BATCH EVALUATION
    # =========================================================

    def evaluate_traces(
        self,
        *,
        trace_ids: list[str],
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    ) -> dict[str, Any]:
        """
        Evaluate multiple existing Langfuse traces.

        Invalid/unmapped traces are skipped.

        No production query is rerun.
        """

        golden_dataset = (
            self.load_golden_dataset()
        )

        golden_by_question = (
            self._golden_by_question(
                golden_dataset
            )
        )

        trace_reviews = (
            self.load_trace_source_review()
        )

        trace_reviews_by_id = (
            self._trace_review_by_id(
                trace_reviews
            )
        )

        processed: list[
            dict[str, Any]
        ] = []

        skipped: list[
            dict[str, str]
        ] = []

        normalized_trace_ids: list[str] = []

        seen_trace_ids: set[str] = set()

        # -----------------------------------------------------
        # Normalize and deduplicate trace IDs.
        # -----------------------------------------------------

        for trace_id in trace_ids:
            normalized = trace_id.strip()

            if not normalized:
                continue

            if normalized in seen_trace_ids:
                continue

            seen_trace_ids.add(
                normalized
            )

            normalized_trace_ids.append(
                normalized
            )

        # -----------------------------------------------------
        # Process each existing trace independently.
        # -----------------------------------------------------

        for trace_id in normalized_trace_ids:
            try:
                result = self.evaluate_trace(
                    trace_id=trace_id,
                    golden_by_question=(
                        golden_by_question
                    ),
                    trace_reviews_by_id=(
                        trace_reviews_by_id
                    ),
                    k_values=k_values,
                )

                processed.append(
                    result
                )

            except Exception as exc:
                skipped.append(
                    {
                        "trace_id": trace_id,
                        "reason": str(exc),
                    }
                )

        # -----------------------------------------------------
        # Aggregate metrics.
        # -----------------------------------------------------

        aggregate = (
            self._calculate_aggregate(
                processed=processed,
                k_values=k_values,
            )
        )

        output = {
            "evaluation_type": (
                "existing_langfuse_trace_retrieval"
            ),
            "rerun_queries": False,
            "llm_calls": False,
            "total_trace_ids": len(
                normalized_trace_ids
            ),
            "processed_count": len(
                processed
            ),
            "skipped_count": len(
                skipped
            ),
            "processed": processed,
            "skipped": skipped,
            "aggregate": aggregate,
        }

        self.save(
            output
        )

        return output

    # =========================================================
    # AGGREGATION
    # =========================================================

    @staticmethod
    def _calculate_aggregate(
        *,
        processed: list[dict[str, Any]],
        k_values: tuple[int, ...],
    ) -> dict[str, Any]:
        """
        Aggregate per-trace retrieval metrics.
        """

        count = len(
            processed
        )

        if count == 0:
            return {
                "evaluated_examples": 0,
                "recall_at_k": {
                    str(k): 0.0
                    for k in k_values
                },
                "precision_at_k": {
                    str(k): 0.0
                    for k in k_values
                },
                "mrr": 0.0,
                "ndcg_at_k": {
                    str(k): 0.0
                    for k in k_values
                },
            }

        recall_at_k: dict[
            str,
            float,
        ] = {}

        precision_at_k: dict[
            str,
            float,
        ] = {}

        ndcg_at_k: dict[
            str,
            float,
        ] = {}

        for k in k_values:
            key = str(k)

            recall_at_k[key] = (
                sum(
                    float(
                        item[
                            "recall_at_k"
                        ][key]
                    )
                    for item in processed
                )
                / count
            )

            precision_at_k[key] = (
                sum(
                    float(
                        item[
                            "precision_at_k"
                        ][key]
                    )
                    for item in processed
                )
                / count
            )

            ndcg_at_k[key] = (
                sum(
                    float(
                        item[
                            "ndcg_at_k"
                        ][key]
                    )
                    for item in processed
                )
                / count
            )

        mrr = (
            sum(
                float(
                    item["mrr"]
                )
                for item in processed
            )
            / count
        )

        return {
            "evaluated_examples": count,
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "mrr": mrr,
            "ndcg_at_k": ndcg_at_k,
        }

    # =========================================================
    # OUTPUT
    # =========================================================

    def save(
        self,
        payload: dict[str, Any],
    ) -> None:
        """
        Save retrieval evaluation results.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_file.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )