from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.existing_trace_langfuse_publisher import (
    ExistingTraceLangfusePublisher,
)
from app.observability.langfuse import configure_langfuse


BASE_DIR = Path(__file__).resolve().parent

RETRIEVAL_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_retrieval_evaluation.json"
)

LLM_JUDGE_FILE = (
    BASE_DIR
    / "datasets"
    / "contextops_llm_judge_evaluation.json"
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Evaluation file must contain "
            f"a JSON object: {path}"
        )

    return payload


def _index_results(
    payload: dict[str, Any],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    results = payload.get(
        field_name,
        []
    )

    if field_name == "processed":
        items = results
    else:
        items = results

    if not isinstance(
        items,
        list,
    ):
        raise ValueError(
            f"{field_name} must be a list."
        )

    indexed: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        trace_id = item.get(
            "trace_id"
        )

        if not isinstance(
            trace_id,
            str,
        ):
            continue

        trace_id = trace_id.strip()

        if trace_id:
            indexed[trace_id] = item

    return indexed


def main() -> None:
    print()
    print("=" * 60)
    print(
        "CONTEXTOPS LANGFUSE EXISTING-TRACE "
        "EVALUATION PUBLISHER"
    )
    print("=" * 60)

    if not configure_langfuse():
        raise RuntimeError(
            "Langfuse is not configured."
        )

    retrieval_payload = _load_json(
        RETRIEVAL_FILE
    )

    judge_payload = _load_json(
        LLM_JUDGE_FILE
    )

    retrieval_results = _index_results(
        retrieval_payload,
        "processed",
    )

    judge_results = _index_results(
        judge_payload,
        "results",
    )

    trace_ids = sorted(
        set(retrieval_results)
        & set(judge_results)
    )

    print(
        f"Retrieval results: {len(retrieval_results)}"
    )

    print(
        f"LLM-judge results: {len(judge_results)}"
    )

    print(
        f"Matching traces:  {len(trace_ids)}"
    )

    if not trace_ids:
        raise RuntimeError(
            "No matching trace IDs were found "
            "between the retrieval and LLM-judge "
            "evaluation artifacts."
        )

    publisher = (
        ExistingTraceLangfusePublisher()
    )

    total_scores = 0
    published_traces = 0

    try:
        for trace_id in trace_ids:
            retrieval_result = (
                retrieval_results[trace_id]
            )

            judge_result = (
                judge_results[trace_id]
            )

            count = publisher.publish_trace(
                retrieval_result=retrieval_result,
                llm_judge_result=judge_result,
            )

            total_scores += count
            published_traces += 1

            print()
            print(
                f"Trace: {trace_id}"
            )
            print(
                f"Question: "
                f"{judge_result.get('question')}"
            )
            print(
                f"Scores published: {count}"
            )

    finally:
        client = publisher.client

        if client is not None:
            flush = getattr(
                client,
                "flush",
                None,
            )

            if callable(flush):
                flush()

    print()
    print("=" * 60)
    print("LANGFUSE PUBLISH COMPLETE")
    print("=" * 60)
    print(
        f"Published traces: {published_traces}"
    )
    print(
        f"Published scores: {total_scores}"
    )
    print(
        "Query rerun:      False"
    )
    print(
        "Retrieval rerun:  False"
    )
    print(
        "Judge rerun:      False"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()