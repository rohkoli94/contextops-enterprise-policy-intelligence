from app.evaluation.models import (
    EvaluationExample,
    EvaluationResult,
    LLMJudgeEvaluation,
)


def test_evaluation_example_defaults() -> None:
    example = EvaluationExample(
        question="What is the leave policy?",
        tenant_id="tenant-001",
    )

    assert example.expected_answer is None
    assert example.expected_source_ids == []
    assert example.conversation_id is None
    assert example.filters is None


def test_evaluation_result_defaults() -> None:
    result = EvaluationResult(
        question="What is the leave policy?",
        answer="Annual leave is available.",
        expected_answer=None,
        retrieval_confidence="strong",
        retrieval_score=0.9,
        retrieval_sufficient=True,
        grounding_status="grounded",
        context_token_count=100,
        latency_ms=200.0,
        citation_count=1,
    )

    assert result.metadata == {}
    assert result.judge_evaluation is None


def test_llm_judge_evaluation_defaults_evaluator() -> None:
    evaluation = LLMJudgeEvaluation(
        correctness_score=4.0,
        grounding_score=4.0,
        citation_score=4.0,
        reason="Excellent.",
    )

    assert evaluation.evaluator == (
        "llm_judge"
    )