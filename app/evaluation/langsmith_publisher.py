import logging
from typing import Any

from langsmith import Client

from app.config.settings import settings
from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.models import EvaluationResult


logger = logging.getLogger(
    "contextops.evaluation.langsmith"
)


class LangSmithEvaluationPublisher:
    """
    Publish ContextOps evaluation results to LangSmith.

    Responsibilities:

        - create the LangSmith client
        - associate evaluation feedback with the
          corresponding LangSmith run
        - publish LLM-judge scores
        - publish aggregate evaluation metrics

    Evaluation remains outside the production LangGraph.
    """

    def __init__(
        self,
        *,
        client: Client | None = None,
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
    def _create_client() -> Client | None:
        """
        Create a LangSmith client when tracing is configured.

        Returns None when LangSmith is intentionally disabled
        or credentials are unavailable.
        """

        if not settings.langsmith_tracing:
            logger.info(
                "LangSmith evaluation publishing is disabled."
            )
            return None

        if not settings.langsmith_api_key:
            logger.warning(
                "LangSmith evaluation publishing is enabled "
                "but LANGSMITH_API_KEY is not configured."
            )
            return None

        return Client(
            api_url=settings.langsmith_endpoint,
            api_key=settings.langsmith_api_key,
        )

    # =========================================================
    # RUN ID
    # =========================================================

    @staticmethod
    def _get_run_id(
        result: EvaluationResult,
    ) -> str | None:
        """
        Extract the LangSmith run identifier from an
        EvaluationResult.
        """

        run_id = result.metadata.get(
            "langsmith_run_id"
        )

        if not isinstance(
            run_id,
            str,
        ):
            return None

        normalized_run_id = run_id.strip()

        return (
            normalized_run_id
            if normalized_run_id
            else None
        )

    # =========================================================
    # SINGLE RESULT
    # =========================================================

    def publish_result(
        self,
        *,
        result: EvaluationResult,
    ) -> bool:
        """
        Publish one evaluation result to its LangSmith run.

        The LangSmith run ID is obtained from:

            result.metadata["langsmith_run_id"]

        Returns:

            True:
                feedback published successfully.

            False:
                publishing was skipped or failed.
        """

        if self.client is None:
            return False

        run_id = self._get_run_id(
            result
        )

        if run_id is None:
            logger.warning(
                "Skipping LangSmith evaluation feedback "
                "because langsmith_run_id is missing.",
                extra={
                    "question": result.question,
                },
            )
            return False

        evaluation = (
            result.judge_evaluation
        )

        if evaluation is None:
            logger.debug(
                "Skipping LangSmith evaluation feedback "
                "because no LLM-judge result exists.",
                extra={
                    "run_id": run_id,
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
            # -------------------------------------------------
            # Correctness
            # -------------------------------------------------

            self.client.create_feedback(
                run_id=run_id,
                key="contextops_correctness",
                score=(
                    evaluation.correctness_score
                ),
                comment=evaluation.reason,
                metadata={
                    **base_metadata,
                    "metric": "correctness",
                },
            )

            # -------------------------------------------------
            # Grounding
            # -------------------------------------------------

            self.client.create_feedback(
                run_id=run_id,
                key="contextops_grounding",
                score=(
                    evaluation.grounding_score
                ),
                metadata={
                    **base_metadata,
                    "metric": "grounding",
                },
            )

            # -------------------------------------------------
            # Citations
            # -------------------------------------------------

            self.client.create_feedback(
                run_id=run_id,
                key="contextops_citations",
                score=(
                    evaluation.citation_score
                ),
                metadata={
                    **base_metadata,
                    "metric": "citations",
                },
            )

            logger.info(
                "Published ContextOps LLM-judge "
                "evaluation to LangSmith.",
                extra={
                    "run_id": run_id,
                },
            )

            return True

        except Exception:
            logger.exception(
                "Failed to publish ContextOps evaluation "
                "feedback to LangSmith.",
                extra={
                    "run_id": run_id,
                    "question": result.question,
                },
            )

            return False

    # =========================================================
    # BATCH RESULTS
    # =========================================================

    def publish_results(
        self,
        *,
        results: list[EvaluationResult],
    ) -> int:
        """
        Publish all available LLM-judge results.

        Returns:
            Number of results successfully published.
        """

        published_count = 0

        for result in results:
            if self.publish_result(
                result=result
            ):
                published_count += 1

        return published_count

    # =========================================================
    # AGGREGATE METRICS
    # =========================================================

    def publish_metrics(
        self,
        *,
        metrics: EvaluationMetrics,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Publish aggregate evaluation metrics against a
        representative LangSmith run.

        The aggregate metrics are stored in one feedback item.

        Args:
            metrics:
                Aggregate ContextOps evaluation metrics.

            run_id:
                LangSmith run ID associated with the evaluation
                execution.

            metadata:
                Optional additional metadata.
        """

        if self.client is None:
            return False

        if not isinstance(
            run_id,
            str,
        ):
            logger.warning(
                "Skipping aggregate LangSmith metrics "
                "because run_id is invalid."
            )
            return False

        normalized_run_id = run_id.strip()

        if not normalized_run_id:
            logger.warning(
                "Skipping aggregate LangSmith metrics "
                "because run_id is empty."
            )
            return False

        aggregate_metadata: dict[str, Any] = {
            "application": "contextops",
            "environment": settings.environment,
            "evaluator": "contextops_evaluation",
            "total_examples": (
                metrics.total_examples
            ),
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
            self.client.create_feedback(
                run_id=normalized_run_id,
                key="contextops_evaluation_summary",
                score=(
                    metrics.average_correctness_score
                    / 4.0
                ),
                metadata=aggregate_metadata,
            )

            logger.info(
                "Published ContextOps aggregate "
                "evaluation metrics to LangSmith.",
                extra={
                    "run_id": normalized_run_id,
                    "total_examples": (
                        metrics.total_examples
                    ),
                },
            )

            return True

        except Exception:
            logger.exception(
                "Failed to publish aggregate ContextOps "
                "evaluation metrics to LangSmith.",
                extra={
                    "run_id": normalized_run_id,
                },
            )

            return False