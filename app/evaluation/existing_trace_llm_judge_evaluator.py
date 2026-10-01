from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.langfuse_golden_dataset_service import (
    LangfuseGoldenDatasetService,
)
from app.evaluation.llm_judge import (
    LLMJudge,
)
from app.evaluation.models import (
    EvaluationResult,
)
from app.observability.langfuse import (
    configure_langfuse,
)
from app.providers.llm.microsoft_foundry import (
    MicrosoftFoundryProvider,
)


BASE_DIR = Path(__file__).resolve().parent

GOLDEN_DATASET_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_golden.json"
)

TRACE_SOURCE_REVIEW_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_trace_source_review.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_llm_judge_evaluation.json"
)


class ExistingTraceLLMJudgeEvaluator:
    """
    Run LLM-as-a-judge evaluation against already-existing
    Langfuse traces.

    IMPORTANT:

    This evaluator does NOT:
        - call QueryService
        - execute /query
        - perform retrieval
        - call Qdrant
        - rerun LangGraph
        - regenerate the original answer

    It reads the existing trace and sends only the existing
    evaluation material to the LLM judge:

        question
        expected answer
        observed answer
        retrieved context
        generated citations
    """

    def __init__(
        self,
        *,
        langfuse_service: (
            LangfuseGoldenDatasetService | None
        ) = None,
        judge: LLMJudge | None = None,
        golden_dataset_file: Path | None = None,
        trace_source_review_file: Path | None = None,
        output_file: Path | None = None,
    ) -> None:
        configured = configure_langfuse()

        if not configured:
            raise RuntimeError(
                "Langfuse is not configured. "
                "Configure LANGFUSE_PUBLIC_KEY, "
                "LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL."
            )

        self.langfuse_service = (
            langfuse_service
            or LangfuseGoldenDatasetService()
        )

        self.judge = (
            judge
            or LLMJudge(
                llm_provider=MicrosoftFoundryProvider()
            )
        )

        self.golden_dataset_file = (
            golden_dataset_file
            or GOLDEN_DATASET_FILE
        )

        self.trace_source_review_file = (
            trace_source_review_file
            or TRACE_SOURCE_REVIEW_FILE
        )

        self.output_file = (
            output_file
            or OUTPUT_FILE
        )

    # ============================================================
    # DATASET LOADING
    # ============================================================

    def load_golden_dataset(
        self,
    ) -> list[dict[str, Any]]:
        if not self.golden_dataset_file.exists():
            raise FileNotFoundError(
                "Golden dataset not found: "
                f"{self.golden_dataset_file}"
            )

        payload = json.loads(
            self.golden_dataset_file.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, list):
            raise ValueError(
                "Golden dataset must contain a JSON array."
            )

        return payload

    def load_trace_source_review(
        self,
    ) -> list[dict[str, Any]]:
        if not self.trace_source_review_file.exists():
            raise FileNotFoundError(
                "Trace source review file not found: "
                f"{self.trace_source_review_file}"
            )

        payload = json.loads(
            self.trace_source_review_file.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, list):
            raise ValueError(
                "Trace source review must contain a JSON array."
            )

        return payload

    @staticmethod
    def _golden_by_question(
        golden_dataset: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[
            str,
            dict[str, Any],
        ] = {}

        for example in golden_dataset:
            question = example.get("question")

            if not isinstance(question, str):
                continue

            normalized = question.strip().lower()

            if normalized:
                indexed[normalized] = example

        return indexed

    @staticmethod
    def _reviews_by_trace_id(
        reviews: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[
            str,
            dict[str, Any],
        ] = {}

        for review in reviews:
            trace_id = review.get("trace_id")

            if not isinstance(trace_id, str):
                continue

            trace_id = trace_id.strip()

            if trace_id:
                indexed[trace_id] = review

        return indexed

    # ============================================================
    # EXPECTED ANSWER
    # ============================================================

    def _resolve_expected_answer(
        self,
        *,
        question: str,
        golden_by_question: dict[
            str,
            dict[str, Any],
        ],
    ) -> str:
        golden_example = golden_by_question.get(
            question.strip().lower()
        )

        if golden_example is None:
            raise ValueError(
                "No golden dataset example matches "
                f"trace question: {question!r}"
            )

        expected_answer = golden_example.get(
            "expected_answer"
        )

        if not isinstance(
            expected_answer,
            str,
        ) or not expected_answer.strip():
            raise ValueError(
                "Expected answer is not mapped for "
                f"question: {question!r}"
            )

        return expected_answer.strip()

    # ============================================================
    # CONTEXT
    # ============================================================

    @staticmethod
    def _extract_context(
        candidate: dict[str, Any],
    ) -> str:
        context = candidate.get(
            "context",
            {},
        )

        if not isinstance(context, dict):
            return ""

        content = context.get("content")

        if content is None:
            return ""

        return str(content)

    # ============================================================
    # EVALUATION RESULT
    # ============================================================

    @staticmethod
    def _build_evaluation_result(
        *,
        candidate: dict[str, Any],
        expected_answer: str,
    ) -> EvaluationResult:
        retrieval = candidate.get(
            "retrieval",
            {},
        )

        grounding = candidate.get(
            "grounding",
            {},
        )

        context = candidate.get(
            "context",
            {},
        )

        timings = candidate.get(
            "timings",
            {},
        )

        citations = candidate.get(
            "citations",
            [],
        )

        if not isinstance(
            retrieval,
            dict,
        ):
            retrieval = {}

        if not isinstance(
            grounding,
            dict,
        ):
            grounding = {}

        if not isinstance(
            context,
            dict,
        ):
            context = {}

        if not isinstance(
            timings,
            dict,
        ):
            timings = {}

        if not isinstance(
            citations,
            list,
        ):
            citations = []

        latency_ms = timings.get(
            "total_ms"
        )

        if latency_ms is None:
            latency_ms = timings.get(
                "total_latency_ms"
            )

        context_token_count = (
            context.get("token_count")
            or 0
        )

        return EvaluationResult(
            question=str(
                candidate.get("question")
                or ""
            ),
            answer=str(
                candidate.get("observed_answer")
                or ""
            ),
            expected_answer=expected_answer,
            retrieval_confidence=(
                retrieval.get(
                    "retrieval_confidence"
                )
            ),
            retrieval_score=(
                retrieval.get(
                    "retrieval_score"
                )
            ),
            retrieval_sufficient=bool(
                retrieval.get(
                    "retrieval_sufficient"
                )
            ),
            grounding_status=(
                grounding.get("status")
            ),
            context_token_count=int(
                context_token_count
            ),
            latency_ms=(
                float(latency_ms)
                if latency_ms is not None
                else None
            ),
            citation_count=len(citations),
            metadata={
                "trace_id": candidate.get(
                    "trace_id"
                ),
                "tenant_id": candidate.get(
                    "tenant_id"
                ),
                "conversation_id": candidate.get(
                    "conversation_id"
                ),
                "langfuse_trace_id": candidate.get(
                    "trace_id"
                ),
                "evaluation_mode": (
                    "existing_trace"
                ),
                "query_rerun": False,
                "retrieval_rerun": False,
            },
        )

    # ============================================================
    # SINGLE TRACE
    # ============================================================

    async def evaluate_trace(
        self,
        *,
        trace_id: str,
        golden_by_question: dict[
            str,
            dict[str, Any],
        ],
    ) -> dict[str, Any]:
        trace_id = trace_id.strip()

        if not trace_id:
            raise ValueError(
                "trace_id cannot be empty."
            )

        candidate = (
            self.langfuse_service.build_from_trace(
                trace_id=trace_id
            )
        )

        question = candidate.get(
            "question"
        )

        if not isinstance(
            question,
            str,
        ) or not question.strip():
            raise ValueError(
                f"Trace {trace_id} does not contain "
                "a valid question."
            )

        expected_answer = (
            self._resolve_expected_answer(
                question=question,
                golden_by_question=golden_by_question,
            )
        )

        evaluation_result = (
            self._build_evaluation_result(
                candidate=candidate,
                expected_answer=expected_answer,
            )
        )

        context = self._extract_context(
            candidate
        )

        citations = candidate.get(
            "citations",
            [],
        )

        if not isinstance(
            citations,
            list,
        ):
            citations = []

        judge_result = await self.judge.evaluate(
            evaluation_result=evaluation_result,
            context=context,
            citations=citations,
        )

        return {
            "trace_id": trace_id,
            "question": question,
            "expected_answer": expected_answer,
            "observed_answer": (
                candidate.get(
                    "observed_answer"
                )
            ),
            "correctness_score": (
                judge_result.correctness_score
            ),
            "grounding_score": (
                judge_result.grounding_score
            ),
            "citation_score": (
                judge_result.citation_score
            ),
            "reason": judge_result.reason,
            "judge_metadata": judge_result.metadata,
            "retrieval": candidate.get(
                "retrieval",
                {},
            ),
            "grounding": candidate.get(
                "grounding",
                {},
            ),
            "citation_count": len(
                citations
            ),
            "evaluation_mode": (
                "existing_trace"
            ),
            "query_rerun": False,
            "retrieval_rerun": False,
        }

    # ============================================================
    # BATCH
    # ============================================================

    async def evaluate_traces(
        self,
        trace_ids: list[str],
    ) -> list[dict[str, Any]]:
        golden_dataset = (
            self.load_golden_dataset()
        )

        golden_by_question = (
            self._golden_by_question(
                golden_dataset
            )
        )

        results: list[dict[str, Any]] = []

        for trace_id in trace_ids:
            results.append(
                await self.evaluate_trace(
                    trace_id=trace_id,
                    golden_by_question=(
                        golden_by_question
                    ),
                )
            )

        return results

    # ============================================================
    # AGGREGATES
    # ============================================================

    @staticmethod
    def calculate_aggregate_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not results:
            return {
                "total_examples": 0,
                "average_correctness_score": None,
                "average_grounding_score": None,
                "average_citation_score": None,
            }

        return {
            "total_examples": len(results),
            "average_correctness_score": (
                sum(
                    float(
                        result[
                            "correctness_score"
                        ]
                    )
                    for result in results
                )
                / len(results)
            ),
            "average_grounding_score": (
                sum(
                    float(
                        result[
                            "grounding_score"
                        ]
                    )
                    for result in results
                )
                / len(results)
            ),
            "average_citation_score": (
                sum(
                    float(
                        result[
                            "citation_score"
                        ]
                    )
                    for result in results
                )
                / len(results)
            ),
        }

    # ============================================================
    # SAVE
    # ============================================================

    def save(
        self,
        *,
        results: list[dict[str, Any]],
    ) -> None:
        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "evaluation_mode": (
                "existing_trace"
            ),
            "query_rerun": False,
            "retrieval_rerun": False,
            "judge_calls": len(results),
            "aggregate": (
                self.calculate_aggregate_metrics(
                    results
                )
            ),
            "results": results,
        }

        self.output_file.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )