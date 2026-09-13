from unittest.mock import Mock

from app.evaluation.langsmith_publisher import (
    LangSmithEvaluationPublisher,
)
from app.evaluation.metrics import (
    EvaluationMetrics,
)
from app.evaluation.models import (
    EvaluationResult,
    LLMJudgeEvaluation,
)


def _result(
    run_id: str | None = "run-001",
) -> EvaluationResult:
    return EvaluationResult(
        question="What is the leave policy?",
        answer="Employees receive annual leave.",
        expected_answer=(
            "Employees receive annual leave."
        ),
        retrieval_confidence="strong",
        retrieval_score=0.9,
        retrieval_sufficient=True,
        grounding_status="grounded",
        context_token_count=100,
        latency_ms=200.0,
        citation_count=1,
        metadata={
            "tenant_id": "tenant-001",
            "langsmith_run_id": run_id,
        },
        judge_evaluation=LLMJudgeEvaluation(
            correctness_score=4.0,
            grounding_score=4.0,
            citation_score=3.0,
            reason="Strong answer.",
        ),
    )


def _metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        total_examples=2,
        retrieval_hit_rate=0.5,
        grounding_rate=1.0,
        citation_coverage=1.0,
        average_latency_ms=200.0,
        cache_hit_rate=0.5,
        retrieval_confidence_rate=1.0,
        average_correctness_score=3.5,
        average_judge_grounding_score=4.0,
        average_judge_citation_score=3.5,
        judge_evaluation_coverage=1.0,
    )


def test_publish_result_creates_three_feedback_items() -> None:
    client = Mock()

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    result = publisher.publish_result(
        result=_result()
    )

    assert result is True

    assert client.create_feedback.call_count == 3

    keys = [
        call.kwargs["key"]
        for call in (
            client.create_feedback.call_args_list
        )
    ]

    assert keys == [
        "contextops_correctness",
        "contextops_grounding",
        "contextops_citations",
    ]


def test_publish_result_skips_missing_run_id() -> None:
    client = Mock()

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    result = publisher.publish_result(
        result=_result(run_id=None)
    )

    assert result is False
    client.create_feedback.assert_not_called()


def test_publish_result_skips_missing_judge_result() -> None:
    client = Mock()

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    evaluation_result = _result()

    evaluation_result = EvaluationResult(
        question=evaluation_result.question,
        answer=evaluation_result.answer,
        expected_answer=(
            evaluation_result.expected_answer
        ),
        retrieval_confidence=(
            evaluation_result.retrieval_confidence
        ),
        retrieval_score=(
            evaluation_result.retrieval_score
        ),
        retrieval_sufficient=(
            evaluation_result.retrieval_sufficient
        ),
        grounding_status=(
            evaluation_result.grounding_status
        ),
        context_token_count=(
            evaluation_result.context_token_count
        ),
        latency_ms=evaluation_result.latency_ms,
        citation_count=evaluation_result.citation_count,
        metadata=evaluation_result.metadata,
        judge_evaluation=None,
    )

    result = publisher.publish_result(
        result=evaluation_result
    )

    assert result is False
    client.create_feedback.assert_not_called()


def test_publish_results_returns_success_count() -> None:
    client = Mock()

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    results = publisher.publish_results(
        results=[
            _result("run-001"),
            _result("run-002"),
        ]
    )

    assert results == 2
    assert client.create_feedback.call_count == 6


def test_publish_metrics_creates_summary_feedback() -> None:
    client = Mock()

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    result = publisher.publish_metrics(
        metrics=_metrics(),
        run_id="run-001",
        metadata={
            "dataset": "golden",
        },
    )

    assert result is True

    client.create_feedback.assert_called_once()

    call = (
        client.create_feedback.call_args
    )

    assert call.kwargs["run_id"] == (
        "run-001"
    )

    assert call.kwargs["key"] == (
        "contextops_evaluation_summary"
    )

    assert call.kwargs["score"] == (
        3.5 / 4.0
    )

    assert call.kwargs["metadata"][
        "dataset"
    ] == "golden"

    assert call.kwargs["metadata"][
        "retrieval_hit_rate"
    ] == 0.5


def test_publish_metrics_skips_empty_run_id() -> None:
    client = Mock()

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    result = publisher.publish_metrics(
        metrics=_metrics(),
        run_id="   ",
    )

    assert result is False
    client.create_feedback.assert_not_called()


def test_publish_result_handles_client_failure() -> None:
    client = Mock()

    client.create_feedback.side_effect = (
        RuntimeError(
            "LangSmith unavailable"
        )
    )

    publisher = (
        LangSmithEvaluationPublisher(
            client=client
        )
    )

    result = publisher.publish_result(
        result=_result()
    )

    assert result is False