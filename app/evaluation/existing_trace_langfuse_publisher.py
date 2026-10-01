from __future__ import annotations

import logging
from typing import Any

from langfuse import get_client

from app.config.settings import settings


logger = logging.getLogger(
    "contextops.evaluation.existing_trace_langfuse"
)


RECALL_SCORES = (
    (1, "contextops_recall_at_1"),
    (3, "contextops_recall_at_3"),
    (5, "contextops_recall_at_5"),
    (10, "contextops_recall_at_10"),
    (20, "contextops_recall_at_20"),
)

PRECISION_SCORES = (
    (1, "contextops_precision_at_1"),
    (3, "contextops_precision_at_3"),
    (5, "contextops_precision_at_5"),
    (10, "contextops_precision_at_10"),
    (20, "contextops_precision_at_20"),
)

NDCG_SCORES = (
    (1, "contextops_ndcg_at_1"),
    (3, "contextops_ndcg_at_3"),
    (5, "contextops_ndcg_at_5"),
    (10, "contextops_ndcg_at_10"),
    (20, "contextops_ndcg_at_20"),
)


class ExistingTraceLangfusePublisher:
    """
    Publish evaluation results against existing Langfuse traces.

    This publisher does not execute the ContextOps query flow.

    It only publishes already-calculated evaluation results:

        Retrieval:
            Recall@K
            Precision@K
            MRR
            NDCG@K

        LLM-as-a-judge:
            Correctness
            Grounding
            Citation

    All scores are attached to the original Langfuse trace ID.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
    ) -> None:
        self.client = (
            client
            if client is not None
            else self._create_client()
        )

    # =========================================================
    # CLIENT
    # =========================================================

    @staticmethod
    def _create_client() -> Any | None:
        if not settings.langfuse_tracing:
            logger.info(
                "Langfuse publishing is disabled."
            )
            return None

        if not settings.langfuse_public_key:
            logger.warning(
                "LANGFUSE_PUBLIC_KEY is not configured."
            )
            return None

        if not settings.langfuse_secret_key:
            logger.warning(
                "LANGFUSE_SECRET_KEY is not configured."
            )
            return None

        return get_client()

    # =========================================================
    # TRACE ID
    # =========================================================

    @staticmethod
    def _normalize_trace_id(
        trace_id: Any,
    ) -> str | None:
        if not isinstance(
            trace_id,
            str,
        ):
            return None

        normalized = trace_id.strip()

        if not normalized:
            return None

        return normalized

    # =========================================================
    # CREATE SCORE
    # =========================================================

    def _create_score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        metric: str,
        metadata: dict[str, Any] | None = None,
        comment: str | None = None,
    ) -> None:
        score_metadata: dict[str, Any] = {
            "application": "contextops",
            "environment": settings.environment,
            "evaluator": (
                "contextops_existing_trace_evaluation"
            ),
            "metric": metric,
        }

        if metadata:
            score_metadata.update(metadata)

        kwargs: dict[str, Any] = {
            "name": name,
            "value": float(value),
            "trace_id": trace_id,
            "data_type": "NUMERIC",
            "metadata": score_metadata,
        }

        if comment:
            kwargs["comment"] = comment

        self.client.create_score(
            **kwargs
        )

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def publish_retrieval_result(
        self,
        *,
        result: dict[str, Any],
    ) -> int:
        """
        Publish all retrieval metrics for one trace.

        Returns the number of scores successfully created.
        """

        if self.client is None:
            return 0

        trace_id = self._normalize_trace_id(
            result.get("trace_id")
        )

        if trace_id is None:
            logger.warning(
                "Skipping retrieval publication because "
                "trace_id is missing."
            )
            return 0

        published = 0

        try:
            recall_at_k = result.get(
                "recall_at_k",
                {},
            )

            for k, score_name in RECALL_SCORES:
                value = recall_at_k.get(
                    str(k)
                )

                if value is None:
                    continue

                self._create_score(
                    trace_id=trace_id,
                    name=score_name,
                    value=float(value),
                    metric="retrieval_recall",
                    metadata={
                        "k": k,
                        "question": result.get(
                            "question"
                        ),
                    },
                )

                published += 1

            precision_at_k = result.get(
                "precision_at_k",
                {},
            )

            for k, score_name in PRECISION_SCORES:
                value = precision_at_k.get(
                    str(k)
                )

                if value is None:
                    continue

                self._create_score(
                    trace_id=trace_id,
                    name=score_name,
                    value=float(value),
                    metric="retrieval_precision",
                    metadata={
                        "k": k,
                        "question": result.get(
                            "question"
                        ),
                    },
                )

                published += 1

            mrr = result.get("mrr")

            if mrr is not None:
                self._create_score(
                    trace_id=trace_id,
                    name="contextops_mrr",
                    value=float(mrr),
                    metric="retrieval_mrr",
                    metadata={
                        "question": result.get(
                            "question"
                        ),
                    },
                )

                published += 1

            ndcg_at_k = result.get(
                "ndcg_at_k",
                {},
            )

            for k, score_name in NDCG_SCORES:
                value = ndcg_at_k.get(
                    str(k)
                )

                if value is None:
                    continue

                self._create_score(
                    trace_id=trace_id,
                    name=score_name,
                    value=float(value),
                    metric="retrieval_ndcg",
                    metadata={
                        "k": k,
                        "question": result.get(
                            "question"
                        ),
                    },
                )

                published += 1

        except Exception:
            logger.exception(
                "Failed to publish retrieval evaluation "
                "scores to Langfuse.",
                extra={
                    "trace_id": trace_id,
                },
            )

            return 0

        return published

    # =========================================================
    # LLM JUDGE
    # =========================================================

    def publish_llm_judge_result(
        self,
        *,
        result: dict[str, Any],
    ) -> int:
        """
        Publish existing LLM-as-a-judge scores for one trace.

        Returns the number of scores successfully created.
        """

        if self.client is None:
            return 0

        trace_id = self._normalize_trace_id(
            result.get("trace_id")
        )

        if trace_id is None:
            logger.warning(
                "Skipping LLM-judge publication because "
                "trace_id is missing."
            )
            return 0

        published = 0

        try:
            judge_scores = (
                (
                    "correctness_score",
                    "contextops_correctness",
                    "correctness",
                ),
                (
                    "grounding_score",
                    "contextops_grounding",
                    "grounding",
                ),
                (
                    "citation_score",
                    "contextops_citations",
                    "citations",
                ),
            )

            for field_name, score_name, metric in (
                judge_scores
            ):
                value = result.get(
                    field_name
                )

                if value is None:
                    continue

                metadata = {
                    "question": result.get(
                        "question"
                    ),
                    "evaluator": (
                        result.get(
                            "judge_metadata",
                            {}
                        ).get(
                            "evaluator",
                            "llm_judge",
                        )
                    ),
                    "max_score": 4,
                }

                self._create_score(
                    trace_id=trace_id,
                    name=score_name,
                    value=float(value),
                    metric=metric,
                    metadata=metadata,
                    comment=(
                        result.get("reason")
                        if metric == "correctness"
                        else None
                    ),
                )

                published += 1

        except Exception:
            logger.exception(
                "Failed to publish LLM-judge evaluation "
                "scores to Langfuse.",
                extra={
                    "trace_id": trace_id,
                },
            )

            return 0

        return published

    # =========================================================
    # TRACE
    # =========================================================

    def publish_trace(
        self,
        *,
        retrieval_result: dict[str, Any],
        llm_judge_result: dict[str, Any],
    ) -> int:
        """
        Publish all evaluation scores for one existing trace.
        """

        retrieval_trace_id = (
            self._normalize_trace_id(
                retrieval_result.get(
                    "trace_id"
                )
            )
        )

        judge_trace_id = (
            self._normalize_trace_id(
                llm_judge_result.get(
                    "trace_id"
                )
            )
        )

        if (
            retrieval_trace_id is None
            or judge_trace_id is None
        ):
            raise ValueError(
                "Both retrieval and LLM-judge results "
                "must contain a valid trace_id."
            )

        if retrieval_trace_id != judge_trace_id:
            raise ValueError(
                "Retrieval and LLM-judge trace IDs do not match: "
                f"{retrieval_trace_id} != {judge_trace_id}"
            )

        retrieval_count = (
            self.publish_retrieval_result(
                result=retrieval_result
            )
        )

        judge_count = (
            self.publish_llm_judge_result(
                result=llm_judge_result
            )
        )

        return retrieval_count + judge_count