from collections.abc import Sequence
from time import perf_counter

from app.api.v1.query.schemas.query_filter import QueryFilter
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.models import (
    EvaluationExample,
    EvaluationResult,
    LLMJudgeEvaluation,
)
from app.services.query_service import QueryService


class EvaluationRunner:
    """
    Execute evaluation examples through the real ContextOps
    query workflow.

    The runner can perform:

        1. deterministic workflow evaluation
        2. optional LLM-as-a-judge evaluation
    """

    def __init__(
        self,
        query_service: QueryService,
        llm_judge: LLMJudge | None = None,
    ) -> None:
        self.query_service = query_service
        self.llm_judge = llm_judge

    # =========================================================
    # PUBLIC API
    # =========================================================

    async def run(
        self,
        examples: Sequence[EvaluationExample],
    ) -> list[EvaluationResult]:
        """
        Execute all evaluation examples sequentially.

        Sequential execution is intentional for the prototype
        because it keeps evaluation runs easier to debug and
        compare.

        Parallel evaluation can be introduced later when the
        dataset becomes larger.
        """

        results: list[EvaluationResult] = []

        for example in examples:
            result = await self._run_example(
                example
            )

            results.append(result)

        return results

    # =========================================================
    # SINGLE EXAMPLE
    # =========================================================

    async def _run_example(
        self,
        example: EvaluationExample,
    ) -> EvaluationResult:
        """
        Execute one evaluation example and optionally run the
        LLM-as-a-judge evaluator.
        """

        started_at = perf_counter()

        filters = (
            QueryFilter(**example.filters)
            if example.filters
            else None
        )

        # -----------------------------------------------------
        # Execute the real production query path.
        # -----------------------------------------------------

        state = await self.query_service.ask(
            question=example.question,
            tenant_id=example.tenant_id,
            conversation_id=example.conversation_id,
            filters=filters,
        )

        measured_latency_ms = (
            perf_counter() - started_at
        ) * 1000

        # -----------------------------------------------------
        # Prefer application-recorded total latency.
        # -----------------------------------------------------

        timings = state.get(
            "timings",
            {},
        )

        latency_ms = (
            timings.get("total_ms")
            if isinstance(
                timings,
                dict,
            )
            else None
        )

        if not isinstance(
            latency_ms,
            (int, float),
        ):
            latency_ms = measured_latency_ms

        # -----------------------------------------------------
        # Normalize answer.
        # -----------------------------------------------------

        answer = state.get(
            "answer",
            "",
        )

        if not isinstance(
            answer,
            str,
        ):
            answer = str(answer)

        # -----------------------------------------------------
        # Normalize citations.
        # -----------------------------------------------------

        citations = state.get(
            "citations",
            [],
        )

        citation_count = (
            len(citations)
            if isinstance(
                citations,
                list,
            )
            else 0
        )

        # -----------------------------------------------------
        # Build deterministic result first.
        # -----------------------------------------------------

        evaluation_result = EvaluationResult(
            question=example.question,
            answer=answer,
            expected_answer=example.expected_answer,
            retrieval_confidence=state.get(
                "retrieval_confidence",
            ),
            retrieval_score=state.get(
                "retrieval_score",
            ),
            retrieval_sufficient=state.get(
                "retrieval_sufficient",
                False,
            ),
            grounding_status=state.get(
                "grounding_status",
            ),
            context_token_count=state.get(
                "context_token_count",
                0,
            ),
            latency_ms=float(
                latency_ms
            ),
            citation_count=citation_count,
            metadata={
                # -------------------------------------------------
                # Evaluation identity
                # -------------------------------------------------

                "tenant_id": example.tenant_id,
                "conversation_id": (
                    example.conversation_id
                ),

                # -------------------------------------------------
                # LangSmith trace correlation
                # -------------------------------------------------

                "langsmith_run_id": state.get(
                    "langsmith_run_id"
                ),

                # -------------------------------------------------
                # Expected evaluation information
                # -------------------------------------------------

                "expected_source_ids": (
                    list(
                        example.expected_source_ids
                    )
                ),

                # -------------------------------------------------
                # Workflow observations
                # -------------------------------------------------

                "cache_hit": state.get(
                    "cache_hit",
                    False,
                ),
                "cache_written": state.get(
                    "cache_written",
                    False,
                ),
                "query_rewritten": state.get(
                    "query_rewritten",
                    False,
                ),
                "context_pii_detected": state.get(
                    "context_pii_detected",
                    False,
                ),
                "context_compressed": state.get(
                    "context_compressed",
                    False,
                ),
                "grounding_reason": state.get(
                    "grounding_reason",
                ),
                "retrieval_reason": state.get(
                    "retrieval_reason",
                ),
                "timings": timings,
            },
        )

        # -----------------------------------------------------
        # Optional LLM-as-a-judge evaluation.
        # -----------------------------------------------------

        if self.llm_judge is None:
            return evaluation_result

        # -----------------------------------------------------
        # Normalize retrieved context.
        # -----------------------------------------------------

        context = state.get(
            "context",
            "",
        )

        if not isinstance(
            context,
            str,
        ):
            context = str(context)

        # -----------------------------------------------------
        # Execute LLM judge.
        # -----------------------------------------------------

        judge_result = await self.llm_judge.evaluate(
            evaluation_result=evaluation_result,
            context=context,
            citations=(
                citations
                if isinstance(
                    citations,
                    list,
                )
                else []
            ),
        )

        # -----------------------------------------------------
        # Convert judge result into our domain model.
        # -----------------------------------------------------

        judge_evaluation = LLMJudgeEvaluation(
            correctness_score=(
                judge_result.correctness_score
            ),
            grounding_score=(
                judge_result.grounding_score
            ),
            citation_score=(
                judge_result.citation_score
            ),
            reason=judge_result.reason,
            evaluator=judge_result.metadata.get(
                "evaluator",
                "llm_judge",
            ),
        )

        # -----------------------------------------------------
        # EvaluationResult is frozen.
        #
        # Therefore create a new instance rather than mutating
        # the existing result.
        # -----------------------------------------------------

        return EvaluationResult(
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
            latency_ms=(
                evaluation_result.latency_ms
            ),
            citation_count=(
                evaluation_result.citation_count
            ),
            metadata={
                **evaluation_result.metadata,
                "judge_evaluator": (
                    judge_result.metadata.get(
                        "evaluator",
                        "llm_judge",
                    )
                ),
            },
            judge_evaluation=judge_evaluation,
        )