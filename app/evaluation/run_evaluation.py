import asyncio
import logging
from pathlib import Path
from typing import Any

from app.dependencies.container import (
    create_query_service,
)
from app.evaluation.dataset_loader import (
    EvaluationDatasetLoader,
)
from app.evaluation.langsmith_publisher import (
    LangSmithEvaluationPublisher,
)
from app.evaluation.llm_judge import (
    LLMJudge,
)
from app.evaluation.metrics import (
    EvaluationMetrics,
    calculate_evaluation_metrics,
)
from app.evaluation.runner import (
    EvaluationRunner,
)
from app.providers.llm.microsoft_foundry import (
    MicrosoftFoundryProvider,
)


logger = logging.getLogger(
    "contextops.evaluation"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATASET_PATH = (
    Path(__file__).parent
    / "datasets"
    / "contextops_golden.json"
)


# ============================================================
# RESOURCE CLEANUP
# ============================================================

async def close_resources(
    resources: list[Any],
) -> None:
    """
    Close long-lived application resources safely.
    """

    for resource in resources:
        try:
            aclose = getattr(
                resource,
                "aclose",
                None,
            )

            if callable(aclose):
                await aclose()
                continue

            close = getattr(
                resource,
                "close",
                None,
            )

            if callable(close):
                result = close()

                if hasattr(
                    result,
                    "__await__",
                ):
                    await result

        except Exception:
            logger.exception(
                "Failed to close evaluation resource.",
                extra={
                    "resource_type": type(
                        resource
                    ).__name__,
                },
            )


# ============================================================
# EVALUATION
# ============================================================

async def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> tuple[
    list[Any],
    EvaluationMetrics,
]:
    """
    Execute the complete ContextOps evaluation pipeline.

    Flow:

        Golden Dataset
            ↓
        Dataset Loader
            ↓
        QueryService
            ↓
        LangGraph
            ↓
        EvaluationRunner
            ↓
        LLM Judge
            ↓
        Evaluation Metrics
            ↓
        LangSmith Publisher
    """

    query_service = None
    judge_llm_provider = None
    publisher = None

    try:
        # =====================================================
        # STEP 1 — LOAD DATASET
        # =====================================================

        loader = EvaluationDatasetLoader()

        examples = loader.load(
            dataset_path
        )

        if not examples:
            raise ValueError(
                "Evaluation dataset contains no examples."
            )

        logger.info(
            "Loaded evaluation dataset.",
            extra={
                "dataset_path": str(
                    dataset_path
                ),
                "example_count": len(
                    examples
                ),
            },
        )

        # =====================================================
        # STEP 2 — CREATE PRODUCTION QUERY SERVICE
        # =====================================================

        query_service = (
            create_query_service()
        )

        # =====================================================
        # STEP 3 — CREATE EVALUATION LLM
        # =====================================================
        #
        # The judge uses the same LLM provider abstraction as
        # the application, but remains outside the production
        # query workflow.
        #

        judge_llm_provider = (
            MicrosoftFoundryProvider()
        )

        judge = LLMJudge(
            llm_provider=judge_llm_provider,
        )

        # =====================================================
        # STEP 4 — CREATE EVALUATION RUNNER
        # =====================================================

        runner = EvaluationRunner(
            query_service=query_service,
            llm_judge=judge,
        )

        # =====================================================
        # STEP 5 — EXECUTE EVALUATION DATASET
        # =====================================================

        results = await runner.run(
            examples
        )

        # =====================================================
        # STEP 6 — CALCULATE AGGREGATE METRICS
        # =====================================================

        metrics = calculate_evaluation_metrics(
            results
        )

        # =====================================================
        # STEP 7 — CREATE LANGSMITH PUBLISHER
        # =====================================================

        publisher = (
            LangSmithEvaluationPublisher()
        )

        # =====================================================
        # STEP 8 — PUBLISH INDIVIDUAL RESULTS
        # =====================================================

        published_count = 0

        for result in results:
            published = (
                publisher.publish_result(
                    result=result,
                )
            )

            if published:
                published_count += 1

        # =====================================================
        # STEP 9 — PUBLISH AGGREGATE METRICS
        # =====================================================
        #
        # A LangSmith feedback item is run-scoped.
        #
        # For the prototype, aggregate metrics are attached to
        # the first evaluation run so the complete experiment
        # remains discoverable from LangSmith.
        #
        # The dedicated experiment/dataset integration can be
        # introduced as the evaluation dataset grows.
        #

        aggregate_published = False

        if results:
            first_run_id = results[0].metadata.get(
                "langsmith_run_id"
            )

            if isinstance(
                first_run_id,
                str,
            ) and first_run_id.strip():
                aggregate_published = (
                    publisher.publish_metrics(
                        metrics=metrics,
                        run_id=first_run_id,
                        metadata={
                            "evaluation_dataset": (
                                str(dataset_path)
                            ),
                        },
                    )
                )

        # =====================================================
        # STEP 10 — LOG SUMMARY
        # =====================================================

        logger.info(
            "ContextOps evaluation completed.",
            extra={
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
                "langsmith_results_published": (
                    published_count
                ),
                "langsmith_aggregate_published": (
                    aggregate_published
                ),
            },
        )

        return results, metrics

    finally:
        # =====================================================
        # CLEANUP JUDGE PROVIDER
        # =====================================================

        if judge_llm_provider is not None:
            try:
                aclose = getattr(
                    judge_llm_provider,
                    "aclose",
                    None,
                )

                if callable(aclose):
                    await aclose()

                else:
                    close = getattr(
                        judge_llm_provider,
                        "close",
                        None,
                    )

                    if callable(close):
                        result = close()

                        if hasattr(
                            result,
                            "__await__",
                        ):
                            await result

            except Exception:
                logger.exception(
                    "Failed to close evaluation "
                    "LLM provider."
                )

        # =====================================================
        # CLEANUP QUERY SERVICE RESOURCES
        # =====================================================

        if query_service is not None:
            resources = list(
                getattr(
                    query_service,
                    "shutdown_resources",
                    [],
                )
            )

            await close_resources(
                resources
            )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(
    metrics: EvaluationMetrics,
) -> None:
    """
    Print a human-readable evaluation summary.
    """

    print()
    print("=" * 60)
    print("ContextOps Evaluation Summary")
    print("=" * 60)

    print(
        f"Total examples:              "
        f"{metrics.total_examples}"
    )

    print(
        f"Retrieval hit rate:          "
        f"{metrics.retrieval_hit_rate:.2%}"
    )

    print(
        f"Grounding rate:              "
        f"{metrics.grounding_rate:.2%}"
    )

    print(
        f"Citation coverage:           "
        f"{metrics.citation_coverage:.2%}"
    )

    print(
        f"Average latency:             "
        f"{metrics.average_latency_ms:.2f} ms"
    )

    print(
        f"Cache hit rate:              "
        f"{metrics.cache_hit_rate:.2%}"
    )

    print(
        f"Retrieval confidence rate:   "
        f"{metrics.retrieval_confidence_rate:.2%}"
    )

    print(
        f"Judge correctness:           "
        f"{metrics.average_correctness_score:.2f}/4"
    )

    print(
        f"Judge grounding:             "
        f"{metrics.average_judge_grounding_score:.2f}/4"
    )

    print(
        f"Judge citations:             "
        f"{metrics.average_judge_citation_score:.2f}/4"
    )

    print(
        f"Judge coverage:              "
        f"{metrics.judge_evaluation_coverage:.2%}"
    )

    print("=" * 60)
    print()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main() -> None:
    """
    Command-line entry point for ContextOps evaluation.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )

    results, metrics = asyncio.run(
        run_evaluation()
    )

    print_summary(
        metrics
    )

    # --------------------------------------------------------
    # Optional detailed result output.
    # --------------------------------------------------------

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"[{index}] "
            f"{result.question}"
        )

        print(
            f"    Retrieval: "
            f"{result.retrieval_confidence}"
        )

        print(
            f"    Grounding: "
            f"{result.grounding_status}"
        )

        print(
            f"    Citations: "
            f"{result.citation_count}"
        )

        print(
            f"    Latency: "
            f"{result.latency_ms:.2f} ms"
        )

        if result.judge_evaluation is not None:
            print(
                f"    Correctness: "
                f"{result.judge_evaluation.correctness_score:.2f}/4"
            )

            print(
                f"    Judge grounding: "
                f"{result.judge_evaluation.grounding_score:.2f}/4"
            )

            print(
                f"    Judge citations: "
                f"{result.judge_evaluation.citation_score:.2f}/4"
            )

        print()


if __name__ == "__main__":
    main()