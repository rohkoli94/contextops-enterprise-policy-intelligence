from __future__ import annotations

import asyncio
from pathlib import Path

from app.evaluation.existing_trace_llm_judge_evaluator import (
    ExistingTraceLLMJudgeEvaluator,
)


BASE_DIR = Path(__file__).resolve().parent

TRACE_IDS_FILE = (
    BASE_DIR
    / "datasets"
    / "retrieval_trace_ids.txt"
)


async def main() -> None:
    if not TRACE_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Trace ID file not found: {TRACE_IDS_FILE}"
        )

    trace_ids = [
        line.strip()
        for line in TRACE_IDS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.strip().startswith("#")
    ]

    if not trace_ids:
        raise ValueError(
            "No trace IDs found."
        )

    evaluator = (
        ExistingTraceLLMJudgeEvaluator()
    )

    results = (
        await evaluator.evaluate_traces(
            trace_ids
        )
    )

    evaluator.save(
        results=results
    )

    aggregate = (
        evaluator.calculate_aggregate_metrics(
            results
        )
    )

    print()
    print("=" * 60)
    print(
        "CONTEXTOPS EXISTING-TRACE LLM JUDGE EVALUATION"
    )
    print("=" * 60)
    print(
        f"Trace IDs:       {len(trace_ids)}"
    )
    print(
        f"Evaluated:       {len(results)}"
    )
    print()
    print(
        "Correctness:"
    )
    print(
        "  Average:       "
        f"{aggregate['average_correctness_score']:.4f}"
    )
    print()
    print(
        "Grounding:"
    )
    print(
        "  Average:       "
        f"{aggregate['average_grounding_score']:.4f}"
    )
    print()
    print(
        "Citation:"
    )
    print(
        "  Average:       "
        f"{aggregate['average_citation_score']:.4f}"
    )
    print()
    print(
        "Output:"
    )
    print(
        "app\\evaluation\\datasets\\"
        "contextops_llm_judge_evaluation.json"
    )
    print()


if __name__ == "__main__":
    asyncio.run(main())