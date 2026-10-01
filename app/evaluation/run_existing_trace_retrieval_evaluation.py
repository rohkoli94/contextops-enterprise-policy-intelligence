from __future__ import annotations

from pathlib import Path

from app.evaluation.existing_trace_retrieval_evaluator import (
    ExistingTraceRetrievalEvaluator,
)


BASE_DIR = Path(__file__).resolve().parent

TRACE_IDS_FILE = (
    BASE_DIR
    / "datasets"
    / "retrieval_trace_ids.txt"
)


def main() -> None:
    if not TRACE_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Trace ID file not found: "
            f"{TRACE_IDS_FILE}"
        )

    trace_ids = [
        line.strip()
        for line in TRACE_IDS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not trace_ids:
        raise ValueError(
            "No Langfuse trace IDs were provided."
        )

    evaluator = (
        ExistingTraceRetrievalEvaluator()
    )

    result = evaluator.evaluate_traces(
        trace_ids=trace_ids
    )

    aggregate = result[
        "aggregate"
    ]

    print(
        "\n"
        "============================================================\n"
        "CONTEXTOPS RETRIEVAL EVALUATION\n"
        "============================================================"
    )

    print(
        f"Trace IDs:       {result['total_trace_ids']}"
    )

    print(
        f"Processed:       {result['processed_count']}"
    )

    print(
        f"Skipped:         {result['skipped_count']}"
    )

    print(
        f"Evaluated:       "
        f"{aggregate['evaluated_examples']}"
    )

    print(
        "\nRecall:"
    )

    for k, value in (
        aggregate["recall_at_k"].items()
    ):
        print(
            f"  Recall@{k}:     {value:.4f}"
        )

    print(
        "\nPrecision:"
    )

    for k, value in (
        aggregate["precision_at_k"].items()
    ):
        print(
            f"  Precision@{k}:  {value:.4f}"
        )

    print(
        f"\nMRR:             "
        f"{aggregate['mrr']:.4f}"
    )

    print(
        "\nNDCG:"
    )

    for k, value in (
        aggregate["ndcg_at_k"].items()
    ):
        print(
            f"  NDCG@{k}:       {value:.4f}"
        )

    print(
        "\nOutput:"
        "\n"
        "app\\evaluation\\datasets\\"
        "contextops_retrieval_evaluation.json"
    )


if __name__ == "__main__":
    main()