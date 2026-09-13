from app.evaluation.metrics import (
    calculate_evaluation_metrics,
)
from app.evaluation.models import (
    EvaluationResult,
    LLMJudgeEvaluation,
)


def _result(
    *,
    retrieval_sufficient: bool = True,
    retrieval_confidence: str | None = "strong",
    grounding_status: str | None = "grounded",
    citation_count: int = 1,
    latency_ms: float = 100.0,
    cache_hit: bool = False,
    judge: LLMJudgeEvaluation | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        question="What is the leave policy?",
        answer="Employees receive annual leave.",
        expected_answer=(
            "Employees receive annual leave."
        ),
        retrieval_confidence=retrieval_confidence,
        retrieval_score=0.9,
        retrieval_sufficient=retrieval_sufficient,
        grounding_status=grounding_status,
        context_token_count=100,
        latency_ms=latency_ms,
        citation_count=citation_count,
        metadata={
            "cache_hit": cache_hit,
        },
        judge_evaluation=judge,
    )


def test_empty_results_return_zero_metrics() -> None:
    metrics = calculate_evaluation_metrics(
        []
    )

    assert metrics.total_examples == 0
    assert metrics.retrieval_hit_rate == 0.0
    assert metrics.grounding_rate == 0.0
    assert metrics.citation_coverage == 0.0
    assert metrics.average_latency_ms == 0.0
    assert metrics.cache_hit_rate == 0.0
    assert metrics.retrieval_confidence_rate == 0.0
    assert metrics.average_correctness_score == 0.0
    assert metrics.average_judge_grounding_score == 0.0
    assert metrics.average_judge_citation_score == 0.0
    assert metrics.judge_evaluation_coverage == 0.0


def test_deterministic_metrics_are_calculated() -> None:
    results = [
        _result(
            retrieval_sufficient=True,
            retrieval_confidence="strong",
            grounding_status="grounded",
            citation_count=2,
            latency_ms=100.0,
            cache_hit=True,
        ),
        _result(
            retrieval_sufficient=False,
            retrieval_confidence="none",
            grounding_status="not_grounded",
            citation_count=0,
            latency_ms=300.0,
            cache_hit=False,
        ),
    ]

    metrics = calculate_evaluation_metrics(
        results
    )

    assert metrics.total_examples == 2
    assert metrics.retrieval_hit_rate == 0.5
    assert metrics.grounding_rate == 0.5
    assert metrics.citation_coverage == 0.5
    assert metrics.average_latency_ms == 200.0
    assert metrics.cache_hit_rate == 0.5
    assert metrics.retrieval_confidence_rate == 1.0


def test_llm_judge_metrics_are_calculated() -> None:
    results = [
        _result(
            judge=LLMJudgeEvaluation(
                correctness_score=4.0,
                grounding_score=3.0,
                citation_score=4.0,
                reason="Excellent.",
            )
        ),
        _result(
            judge=LLMJudgeEvaluation(
                correctness_score=2.0,
                grounding_score=3.0,
                citation_score=2.0,
                reason="Partially correct.",
            )
        ),
        _result(
            judge=None
        ),
    ]

    metrics = calculate_evaluation_metrics(
        results
    )

    assert metrics.judge_evaluation_coverage == (
        2 / 3
    )

    assert metrics.average_correctness_score == 3.0
    assert metrics.average_judge_grounding_score == 3.0
    assert metrics.average_judge_citation_score == 3.0


def test_unknown_retrieval_confidence_is_not_counted() -> None:
    results = [
        _result(
            retrieval_confidence="high"
        ),
        _result(
            retrieval_confidence="strong"
        ),
    ]

    metrics = calculate_evaluation_metrics(
        results
    )

    assert metrics.retrieval_confidence_rate == 0.5