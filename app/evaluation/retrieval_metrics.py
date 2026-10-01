from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any, Iterable, Sequence


DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class RetrievalExampleMetrics:
    """
    Retrieval metrics for one evaluation example.

    expected_source_ids:
        Ground-truth source/chunk IDs.

    retrieved_source_ids:
        Actual ranked source/chunk IDs from the existing trace.
    """

    expected_source_ids: tuple[str, ...]
    retrieved_source_ids: tuple[str, ...]

    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]

    reciprocal_rank: float

    ndcg_at_k: dict[int, float]

    first_relevant_rank: int | None


@dataclass(frozen=True)
class RetrievalAggregateMetrics:
    """
    Aggregate retrieval metrics across the evaluation dataset.
    """

    total_examples: int

    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]

    mean_reciprocal_rank: float

    ndcg_at_k: dict[int, float]

    evaluated_examples: int


def _normalise_ids(values: Iterable[Any]) -> list[str]:
    """
    Convert source IDs to unique strings while preserving order.
    """

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value is None:
            continue

        value_str = str(value).strip()

        if not value_str:
            continue

        if value_str in seen:
            continue

        seen.add(value_str)
        result.append(value_str)

    return result


def _validate_k_values(
    k_values: Sequence[int],
) -> tuple[int, ...]:
    """
    Validate and normalise K values.
    """

    normalised = tuple(sorted(set(int(k) for k in k_values)))

    if not normalised:
        raise ValueError(
            "k_values must contain at least one positive integer."
        )

    if any(k <= 0 for k in normalised):
        raise ValueError(
            "All k_values must be positive integers."
        )

    return normalised


def _recall_at_k(
    expected_ids: set[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """
    Recall@K = relevant retrieved documents in top K /
               total relevant documents.
    """

    if not expected_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])

    hits = len(expected_ids.intersection(top_k))

    return hits / len(expected_ids)


def _precision_at_k(
    expected_ids: set[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """
    Precision@K = relevant retrieved documents in top K / K.

    If fewer than K documents were actually retrieved, the denominator
    is the number of retrieved documents available.
    """

    top_k = list(retrieved_ids[:k])

    if not top_k:
        return 0.0

    hits = sum(
        1
        for source_id in top_k
        if source_id in expected_ids
    )

    return hits / len(top_k)


def _reciprocal_rank(
    expected_ids: set[str],
    retrieved_ids: Sequence[str],
) -> tuple[float, int | None]:
    """
    Reciprocal Rank is 1 / rank of the first relevant result.
    """

    if not expected_ids:
        return 0.0, None

    for rank, source_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if source_id in expected_ids:
            return 1.0 / rank, rank

    return 0.0, None


def _dcg_at_k(
    expected_ids: set[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """
    DCG using binary relevance.

    Relevant source = 1
    Non-relevant source = 0
    """

    score = 0.0

    for rank, source_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if source_id in expected_ids:
            score += 1.0 / log2(rank + 1)

    return score


def _ideal_dcg_at_k(
    relevant_count: int,
    k: int,
) -> float:
    """
    Ideal DCG when all relevant documents occupy
    the highest possible ranks.
    """

    ideal_relevant_count = min(
        relevant_count,
        k,
    )

    score = 0.0

    for rank in range(
        1,
        ideal_relevant_count + 1,
    ):
        score += 1.0 / log2(rank + 1)

    return score


def _ndcg_at_k(
    expected_ids: set[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """
    NDCG@K using binary relevance.
    """

    if not expected_ids:
        return 0.0

    ideal_dcg = _ideal_dcg_at_k(
        relevant_count=len(expected_ids),
        k=k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    dcg = _dcg_at_k(
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        k=k,
    )

    return dcg / ideal_dcg


def calculate_retrieval_metrics(
    expected_source_ids: Sequence[Any],
    retrieved_source_ids: Sequence[Any],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalExampleMetrics:
    """
    Calculate retrieval metrics for one example.

    This function is completely deterministic.

    It does NOT:
        - call QueryService
        - call an LLM
        - perform retrieval
        - access Qdrant
        - modify the golden dataset
    """

    k_values = _validate_k_values(k_values)

    expected_ids_list = _normalise_ids(
        expected_source_ids
    )

    retrieved_ids_list = _normalise_ids(
        retrieved_source_ids
    )

    expected_ids = set(expected_ids_list)

    recall = {
        k: _recall_at_k(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids_list,
            k=k,
        )
        for k in k_values
    }

    precision = {
        k: _precision_at_k(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids_list,
            k=k,
        )
        for k in k_values
    }

    reciprocal_rank, first_relevant_rank = (
        _reciprocal_rank(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids_list,
        )
    )

    ndcg = {
        k: _ndcg_at_k(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids_list,
            k=k,
        )
        for k in k_values
    }

    return RetrievalExampleMetrics(
        expected_source_ids=tuple(
            expected_ids_list
        ),
        retrieved_source_ids=tuple(
            retrieved_ids_list
        ),
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        first_relevant_rank=first_relevant_rank,
    )


def calculate_retrieval_aggregate_metrics(
    examples: Sequence[
        tuple[
            Sequence[Any],
            Sequence[Any],
        ]
    ],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalAggregateMetrics:
    """
    Calculate aggregate retrieval metrics across examples.

    Each example is:

        (
            expected_source_ids,
            retrieved_source_ids,
        )
    """

    k_values = _validate_k_values(k_values)

    if not examples:
        return RetrievalAggregateMetrics(
            total_examples=0,
            recall_at_k={
                k: 0.0
                for k in k_values
            },
            precision_at_k={
                k: 0.0
                for k in k_values
            },
            mean_reciprocal_rank=0.0,
            ndcg_at_k={
                k: 0.0
                for k in k_values
            },
            evaluated_examples=0,
        )

    example_metrics: list[
        RetrievalExampleMetrics
    ] = []

    for expected_ids, retrieved_ids in examples:
        metrics = calculate_retrieval_metrics(
            expected_source_ids=expected_ids,
            retrieved_source_ids=retrieved_ids,
            k_values=k_values,
        )

        example_metrics.append(metrics)

    recall_at_k = {
        k: sum(
            metrics.recall_at_k[k]
            for metrics in example_metrics
        )
        / len(example_metrics)
        for k in k_values
    }

    precision_at_k = {
        k: sum(
            metrics.precision_at_k[k]
            for metrics in example_metrics
        )
        / len(example_metrics)
        for k in k_values
    }

    ndcg_at_k = {
        k: sum(
            metrics.ndcg_at_k[k]
            for metrics in example_metrics
        )
        / len(example_metrics)
        for k in k_values
    }

    mean_reciprocal_rank = (
        sum(
            metrics.reciprocal_rank
            for metrics in example_metrics
        )
        / len(example_metrics)
    )

    return RetrievalAggregateMetrics(
        total_examples=len(examples),
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mean_reciprocal_rank=mean_reciprocal_rank,
        ndcg_at_k=ndcg_at_k,
        evaluated_examples=len(example_metrics),
    )


def extract_retrieved_source_ids(
    retrieved_sources: Sequence[Any],
) -> list[str]:
    """
    Extract ranked chunk IDs from ContextOps trace data.

    Expected ContextOps structure:

        [
            {
                "rank": 1,
                "chunk_id": "...",
                ...
            },
            ...
        ]

    The original trace ranking is preserved.
    """

    source_ids: list[str] = []

    for source in retrieved_sources:
        if isinstance(source, dict):
            chunk_id = source.get(
                "chunk_id"
            )

            if chunk_id is not None:
                source_ids.append(
                    str(chunk_id)
                )

    return _normalise_ids(source_ids)


def calculate_retrieval_metrics_from_trace_candidate(
    candidate: dict[str, Any],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalExampleMetrics:
    """
    Calculate retrieval metrics directly from one
    ContextOps golden-dataset candidate.

    The candidate must contain:

        expected_source_ids
        retrieved_sources
    """

    expected_source_ids = candidate.get(
        "expected_source_ids",
        [],
    )

    retrieved_sources = candidate.get(
        "retrieved_sources",
        [],
    )

    if not isinstance(
        expected_source_ids,
        list,
    ):
        raise ValueError(
            "'expected_source_ids' must be a list."
        )

    if not isinstance(
        retrieved_sources,
        list,
    ):
        raise ValueError(
            "'retrieved_sources' must be a list."
        )

    retrieved_source_ids = (
        extract_retrieved_source_ids(
            retrieved_sources
        )
    )

    return calculate_retrieval_metrics(
        expected_source_ids=expected_source_ids,
        retrieved_source_ids=retrieved_source_ids,
        k_values=k_values,
    )