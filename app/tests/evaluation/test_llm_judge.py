import pytest

from app.evaluation.llm_judge import (
    LLMJudge,
)
from app.evaluation.models import (
    EvaluationResult,
)
from app.providers.llm.base import (
    LLMResponse,
)


def _evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        question="What is the notice period?",
        answer=(
            "The notice period is three months."
        ),
        expected_answer=(
            "The notice period is three months."
        ),
        retrieval_confidence="strong",
        retrieval_score=0.91,
        retrieval_sufficient=True,
        grounding_status="grounded",
        context_token_count=100,
        latency_ms=250.0,
        citation_count=1,
        metadata={},
    )


class FakeLLMProvider:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.received_request = None

    async def agenerate(
        self,
        request,
    ) -> LLMResponse:
        self.received_request = request

        return LLMResponse(
            content=self.response,
            model="judge-model",
            provider="fake-provider",
        )


@pytest.mark.asyncio
async def test_llm_judge_parses_valid_response() -> None:
    provider = FakeLLMProvider(
        response=(
            '{'
            '"correctness_score": 4,'
            '"grounding_score": 4,'
            '"citation_score": 3,'
            '"reason": "Answer is fully supported."'
            '}'
        )
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    result = await judge.evaluate(
        evaluation_result=_evaluation_result(),
        context=(
            "[SOURCE 1]\n"
            "The notice period is three months."
        ),
        citations=[
            {
                "source": 1,
                "chunk_id": "chunk-001",
            }
        ],
    )

    assert result.correctness_score == 4.0
    assert result.grounding_score == 4.0
    assert result.citation_score == 3.0
    assert result.reason == (
        "Answer is fully supported."
    )

    assert result.metadata[
        "evaluator"
    ] == "llm_judge"


@pytest.mark.asyncio
async def test_llm_judge_builds_evaluation_prompt() -> None:
    provider = FakeLLMProvider(
        response=(
            '{'
            '"correctness_score": 4,'
            '"grounding_score": 4,'
            '"citation_score": 4,'
            '"reason": "Good."'
            '}'
        )
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    await judge.evaluate(
        evaluation_result=_evaluation_result(),
        context="The policy says three months.",
        citations=[],
    )

    assert provider.received_request is not None

    prompt = (
        provider.received_request.user_prompt
    )

    assert (
        "What is the notice period?"
        in prompt
    )

    assert (
        "The notice period is three months."
        in prompt
    )

    assert (
        "The policy says three months."
        in prompt
    )


@pytest.mark.asyncio
async def test_llm_judge_rejects_invalid_json() -> None:
    provider = FakeLLMProvider(
        response="not-json"
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        await judge.evaluate(
            evaluation_result=_evaluation_result(),
            context="Some context",
            citations=[],
        )


@pytest.mark.asyncio
async def test_llm_judge_rejects_non_object_json() -> None:
    provider = FakeLLMProvider(
        response="[1, 2, 3]"
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        await judge.evaluate(
            evaluation_result=_evaluation_result(),
            context="Some context",
            citations=[],
        )


def test_llm_judge_rejects_invalid_score() -> None:
    provider = FakeLLMProvider(
        response="{}"
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        judge._parse_score(
            "bad",
            field_name="correctness_score",
        )


def test_llm_judge_rejects_score_above_four() -> None:
    provider = FakeLLMProvider(
        response="{}"
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 4",
    ):
        judge._parse_score(
            5,
            field_name="correctness_score",
        )


def test_llm_judge_rejects_boolean_score() -> None:
    provider = FakeLLMProvider(
        response="{}"
    )

    judge = LLMJudge(
        llm_provider=provider
    )

    with pytest.raises(
        ValueError,
        match="numeric score",
    ):
        judge._parse_score(
            True,
            field_name="correctness_score",
        )