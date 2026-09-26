import logging
from typing import Any

from langfuse import get_client

from app.config.settings import settings
from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.models import EvaluationResult


logger = logging.getLogger(
    "contextops.evaluation.langfuse"
)


class LangfuseEvaluationPublisher:
    """
    Publish ContextOps evaluation results to Langfuse.

    Evaluation remains outside the production LangGraph.
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
        """
        Create or retrieve the configured Langfuse client.

        Returns:
            Configured Langfuse client when tracing credentials
            are available, otherwise None.
        """

        if not settings.langfuse_tracing:
            logger.info(
                "Langfuse evaluation publishing is disabled."
            )
            return None

        if not settings.langfuse_public_key:
            logger.warning(
                "Langfuse evaluation publishing is enabled "
                "but LANGFUSE_PUBLIC_KEY is not configured."
            )
            return None

        if not settings.langfuse_secret_key:
            logger.warning(
                "Langfuse evaluation publishing is enabled "
                "but LANGFUSE_SECRET_KEY is not configured."
            )
            return None

        return get_client()

    # =========================================================
    # TRACE ID
    # =========================================================

    @staticmethod
    def _get_trace_id(
        result: EvaluationResult,
    ) -> str | None:
        """
        Extract the Langfuse trace identifier from an
        EvaluationResult.

        ContextOps stores the identifier using the
        `langfuse_trace_id` metadata key.
        """

        trace_id = result.metadata.get(
            "langfuse_trace_id"
        )

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
    # PUBLISH SINGLE RESULT
    # =========================================================

    def publish_result(
        self,
        *,
        result: EvaluationResult,
    ) -> bool:
        """
        Publish the LLM-as-a-judge evaluation for one result.

        Three Langfuse scores are created:

            1. correctness
            2. grounding
            3. citation

        Returns:
            True when the scores are successfully published.
            False when publishing is skipped or fails.
        """

        if self.client is None:
            return False

        trace_id = self._get_trace_id(
            result
        )

        if trace_id is None:
            logger.warning(
                "Skipping Langfuse evaluation scores because "
                "langfuse_trace_id is missing.",
                extra={
                    "question": result.question,
                },
            )
            return False

        evaluation = result.judge_evaluation

        if evaluation is None:
            logger.debug(
                "Skipping Langfuse evaluation scores because "
                "no LLM-judge result exists.",
                extra={
                    "trace_id": trace_id,
                },
            )
            return False

        base_metadata: dict[str, Any] = {
            "application": "contextops",
            "environment": settings.environment,
            "evaluator": evaluation.evaluator,
            "tenant_id": result.metadata.get(
                "tenant_id"
            ),
        }

        try:
            self.client.create_score(
                name="contextops_correctness",
                value=float(
                    evaluation.correctness_score
                ),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=evaluation.reason,
                metadata={
                    **base_metadata,
                    "metric": "correctness",
                },
            )

            self.client.create_score(
                name="contextops_grounding",
                value=float(
                    evaluation.grounding_score
                ),
                trace_id=trace_id,
                data_type="NUMERIC",
                metadata={
                    **base_metadata,
                    "metric": "grounding",
                },
            )

            self.client.create_score(
                name="contextops_citations",
                value=float(
                    evaluation.citation_score
                ),
                trace_id=trace_id,
                data_type="NUMERIC",
                metadata={
                    **base_metadata,
                    "metric": "citations",
                },
            )

            logger.info(
                "Published ContextOps LLM-judge evaluation "
                "to Langfuse.",
                extra={
                    "trace_id": trace_id,
                },
            )

            return True

        except Exception:
            logger.exception(
                "Failed to publish ContextOps evaluation "
                "scores to Langfuse.",
                extra={
                    "trace_id": trace_id,
                    "question": result.question,
                },
            )

            return False

    # =========================================================
    # PUBLISH MULTIPLE RESULTS
    # =========================================================

    def publish_results(
        self,
        *,
        results: list[EvaluationResult],
    ) -> int:
        """
        Publish all available LLM-as-a-judge evaluations.

        Returns:
            Number of evaluation results successfully
            published.
        """

        published_count = 0

        for result in results:
            if self.publish_result(
                result=result
            ):
                published_count += 1

        return published_count

    # =========================================================
    # PUBLISH AGGREGATE METRICS
    # =========================================================

    def publish_metrics(
        self,
        *,
        metrics: EvaluationMetrics,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Publish aggregate ContextOps evaluation metrics
        against a representative Langfuse trace.

        The aggregate metrics are stored as one Langfuse
        numeric score.

        Args:
            metrics:
                Aggregate ContextOps evaluation metrics.

            trace_id:
                Langfuse trace identifier associated with the
                evaluation execution.

            metadata:
                Optional additional metadata.

        Returns:
            True when the aggregate score is published.
            False when publishing is skipped or fails.
        """

        if self.client is None:
            return False

        if not isinstance(
            trace_id,
            str,
        ):
            logger.warning(
                "Skipping aggregate Langfuse metrics because "
                "trace_id is invalid."
            )
            return False

        normalized_trace_id = trace_id.strip()

        if not normalized_trace_id:
            logger.warning(
                "Skipping aggregate Langfuse metrics because "
                "trace_id is empty."
            )
            return False

        aggregate_metadata: dict[str, Any] = {
            "application": "contextops",
            "environment": settings.environment,
            "evaluator": "contextops_evaluation",
            "total_examples": metrics.total_examples,
            "retrieval_hit_rate": (
                metrics.retrieval_hit_rate
            ),
            "grounding_rate": (
                metrics.grounding_rate
            ),
            "citation_coverage": (
                metrics.citation_coverage
            ),
            "average_latency_ms": (
                metrics.average_latency_ms
            ),
            "cache_hit_rate": (
                metrics.cache_hit_rate
            ),
            "retrieval_confidence_rate": (
                metrics.retrieval_confidence_rate
            ),
            "average_correctness_score": (
                metrics.average_correctness_score
            ),
            "average_judge_grounding_score": (
                metrics.average_judge_grounding_score
            ),
            "average_judge_citation_score": (
                metrics.average_judge_citation_score
            ),
            "judge_evaluation_coverage": (
                metrics.judge_evaluation_coverage
            ),
        }

        if metadata:
            aggregate_metadata.update(
                metadata
            )

        try:
            self.client.create_score(
                name="contextops_evaluation_summary",
                value=float(
                    metrics.average_correctness_score / 4.0
                ),
                trace_id=normalized_trace_id,
                data_type="NUMERIC",
                metadata=aggregate_metadata,
            )

            logger.info(
                "Published ContextOps aggregate "
                "evaluation metrics to Langfuse.",
                extra={
                    "trace_id": normalized_trace_id,
                    "total_examples": (
                        metrics.total_examples
                    ),
                },
            )

            return True

        except Exception:
            logger.exception(
                "Failed to publish aggregate ContextOps "
                "metrics to Langfuse.",
                extra={
                    "trace_id": normalized_trace_id,
                },
            )

            return False