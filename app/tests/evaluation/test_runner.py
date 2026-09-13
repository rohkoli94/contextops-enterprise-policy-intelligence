import pytest

from app.evaluation.llm_judge import (
    LLMJudgeResult,
)
from app.evaluation.models import (
    EvaluationExample,
)
from app.evaluation.runner import (
    EvaluationRunner,
)


class FakeQueryService:
    async def ask(
        self,
        *,
        question,
        tenant_id,
        conversation_id=None,
        filters=None,
    ):
        return {
            "answer": (
                "The notice period is three months. "
                "[SOURCE 1]"
            ),
            "citations": [
                {
                    "source": 1,
                    "chunk_id": "chunk-001",
                }
            ],
            "retrieval_confidence": "strong",
            "retrieval_score": 0.92,
            "retrieval_sufficient": True,
            "grounding_status": "grounded",
            "context_token_count": 120,
            "context": (
                "[SOURCE 1]\n"
                "The notice period is three months."
            ),
            "cache_hit": False,
            "cache_written": True,
            "query_rewritten": True,
            "context_pii_detected": False,
            "context_compressed": True,
            "grounding_reason": "Supported.",
            "retrieval_reason": "Strong evidence.",
            "langsmith_run_id": "run-123",
            "timings": {
                "stages": {
                    "retrieval": 20.5,
                },
                "total_ms": 200.0,
            },
        }


class FakeLLMJudge:
    async def evaluate(
        self,
        *,
        evaluation_result,
        context,
        citations,
    ):
        return LLMJudgeResult(
            correctness_score=4.0,
            grounding_score=4.0,
            citation_score=3.0,
            reason="Strong evaluation.",
            metadata={
                "evaluator": "test_judge",
            },
        )


class FailingLLMJudge:
    async def evaluate(
        self,
        *,
        evaluation_result,
        context,
        citations,
    ):
        raise RuntimeError(
            "Judge failure"
        )


@pytest.mark.asyncio
async def test_runner_without_judge() -> None:
    runner = EvaluationRunner(
        query_service=FakeQueryService()
    )

    results = await runner.run(
        [
            EvaluationExample(
                question="What is the notice period?",
                tenant_id="tenant-001",
                expected_answer=(
                    "The notice period is three months."
                ),
                expected_source_ids=[
                    "chunk-001"
                ],
            )
        ]
    )

    assert len(results) == 1

    result = results[0]

    assert result.answer.startswith(
        "The notice period is three months."
    )

    assert result.citation_count == 1
    assert result.retrieval_sufficient is True
    assert result.grounding_status == "grounded"
    assert result.latency_ms == 200.0

    assert result.metadata[
        "langsmith_run_id"
    ] == "run-123"

    assert result.judge_evaluation is None


@pytest.mark.asyncio
async def test_runner_with_judge() -> None:
    runner = EvaluationRunner(
        query_service=FakeQueryService(),
        llm_judge=FakeLLMJudge(),
    )

    results = await runner.run(
        [
            EvaluationExample(
                question="What is the notice period?",
                tenant_id="tenant-001",
                expected_answer=(
                    "The notice period is three months."
                ),
            )
        ]
    )

    result = results[0]

    assert result.judge_evaluation is not None

    assert (
        result.judge_evaluation.correctness_score
        == 4.0
    )

    assert (
        result.judge_evaluation.grounding_score
        == 4.0
    )

    assert (
        result.judge_evaluation.citation_score
        == 3.0
    )

    assert (
        result.judge_evaluation.evaluator
        == "test_judge"
    )


@pytest.mark.asyncio
async def test_runner_propagates_judge_failure() -> None:
    runner = EvaluationRunner(
        query_service=FakeQueryService(),
        llm_judge=FailingLLMJudge(),
    )

    with pytest.raises(
        RuntimeError,
        match="Judge failure",
    ):
        await runner.run(
            [
                EvaluationExample(
                    question=(
                        "What is the notice period?"
                    ),
                    tenant_id="tenant-001",
                )
            ]
        )