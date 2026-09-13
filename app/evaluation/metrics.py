from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from app.evaluation.models import EvaluationResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Aggregate metrics produced from a set of evaluation results.

    Deterministic metrics:

        - retrieval hit rate
        - grounding rate
        - citation coverage
        - average latency
        - cache hit rate
        - retrieval confidence rate

    LLM-as-a-judge metrics:

        - average correctness score
        - average grounding score
        - average citation score
        - judge evaluation coverage
    """

    total_examples: int

    retrieval_hit_rate: float

    grounding_rate: float

    citation_coverage: float

    average_latency_ms: float

    cache_hit_rate: float

    retrieval_confidence_rate: float

    average_correctness_score: float

    average_judge_grounding_score: float

    average_judge_citation_score: float

    judge_evaluation_coverage: float


def calculate_evaluation_metrics(
    results: Sequence[EvaluationResult],
) -> EvaluationMetrics:
    """
    Calculate aggregate evaluation metrics.

    All rate metrics are returned as values between 0 and 1.

    LLM-judge scores remain on their original 0-4 scale.
    """

    total_examples = len(results)

    if total_examples == 0:
        return EvaluationMetrics(
            total_examples=0,
            retrieval_hit_rate=0.0,
            grounding_rate=0.0,
            citation_coverage=0.0,
            average_latency_ms=0.0,
            cache_hit_rate=0.0,
            retrieval_confidence_rate=0.0,
            average_correctness_score=0.0,
            average_judge_grounding_score=0.0,
            average_judge_citation_score=0.0,
            judge_evaluation_coverage=0.0,
        )

    # =========================================================
    # RETRIEVAL HIT RATE
    # =========================================================

    retrieval_hits = sum(
        1
        for result in results
        if result.retrieval_sufficient
    )

    retrieval_hit_rate = (
        retrieval_hits / total_examples
    )

    # =========================================================
    # DETERMINISTIC GROUNDING RATE
    # =========================================================

    grounded_results = sum(
        1
        for result in results
        if result.grounding_status == "grounded"
    )

    grounding_rate = (
        grounded_results / total_examples
    )

    # =========================================================
    # CITATION COVERAGE
    # =========================================================

    cited_results = sum(
        1
        for result in results
        if result.citation_count > 0
    )

    citation_coverage = (
        cited_results / total_examples
    )

    # =========================================================
    # LATENCY
    # =========================================================

    latencies = [
        result.latency_ms
        for result in results
        if isinstance(
            result.latency_ms,
            (int, float),
        )
    ]

    average_latency_ms = (
        mean(latencies)
        if latencies
        else 0.0
    )

    # =========================================================
    # CACHE HIT RATE
    # =========================================================

    cache_hits = sum(
        1
        for result in results
        if result.metadata.get(
            "cache_hit",
            False,
        )
    )

    cache_hit_rate = (
        cache_hits / total_examples
    )

    # =========================================================
    # RETRIEVAL CONFIDENCE RATE
    # =========================================================

    recognized_confidences = {
        "strong",
        "sufficient",
        "none",
    }

    confidence_results = sum(
        1
        for result in results
        if result.retrieval_confidence
        in recognized_confidences
    )

    retrieval_confidence_rate = (
        confidence_results / total_examples
    )

    # =========================================================
    # LLM-JUDGE RESULTS
    # =========================================================

    judge_results = [
        result.judge_evaluation
        for result in results
        if result.judge_evaluation is not None
    ]

    judge_evaluation_coverage = (
        len(judge_results) / total_examples
    )

    # =========================================================
    # AVERAGE CORRECTNESS
    # =========================================================

    correctness_scores = [
        evaluation.correctness_score
        for evaluation in judge_results
    ]

    average_correctness_score = (
        mean(correctness_scores)
        if correctness_scores
        else 0.0
    )

    # =========================================================
    # AVERAGE JUDGE GROUNDING
    # =========================================================

    judge_grounding_scores = [
        evaluation.grounding_score
        for evaluation in judge_results
    ]

    average_judge_grounding_score = (
        mean(judge_grounding_scores)
        if judge_grounding_scores
        else 0.0
    )

    # =========================================================
    # AVERAGE JUDGE CITATION
    # =========================================================

    judge_citation_scores = [
        evaluation.citation_score
        for evaluation in judge_results
    ]

    average_judge_citation_score = (
        mean(judge_citation_scores)
        if judge_citation_scores
        else 0.0
    )

    return EvaluationMetrics(
        total_examples=total_examples,
        retrieval_hit_rate=retrieval_hit_rate,
        grounding_rate=grounding_rate,
        citation_coverage=citation_coverage,
        average_latency_ms=average_latency_ms,
        cache_hit_rate=cache_hit_rate,
        retrieval_confidence_rate=(
            retrieval_confidence_rate
        ),
        average_correctness_score=(
            average_correctness_score
        ),
        average_judge_grounding_score=(
            average_judge_grounding_score
        ),
        average_judge_citation_score=(
            average_judge_citation_score
        ),
        judge_evaluation_coverage=(
            judge_evaluation_coverage
        ),
    )