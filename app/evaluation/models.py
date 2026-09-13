from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationExample:
    """
    Single evaluation example.

    The example represents the expected behavior for one
    ContextOps question.
    """

    question: str
    tenant_id: str
    expected_answer: str | None = None
    expected_source_ids: list[str] = field(
        default_factory=list
    )
    conversation_id: str | None = None
    filters: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMJudgeEvaluation:
    """
    Semantic evaluation produced by the LLM-as-a-judge.

    Scores use a 0-4 scale.
    """

    correctness_score: float
    grounding_score: float
    citation_score: float
    reason: str
    evaluator: str = "llm_judge"


@dataclass(frozen=True)
class EvaluationResult:
    """
    Result produced by a ContextOps evaluation run.

    The result contains both:

        1. deterministic workflow observations
        2. optional semantic LLM-judge evaluation
    """

    question: str
    answer: str
    expected_answer: str | None

    retrieval_confidence: str | None
    retrieval_score: float | None
    retrieval_sufficient: bool

    grounding_status: str | None

    context_token_count: int
    latency_ms: float | None
    citation_count: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    judge_evaluation: LLMJudgeEvaluation | None = None