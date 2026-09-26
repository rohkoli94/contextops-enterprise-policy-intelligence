from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.golden_source_identifier import (
    GoldenSourceIdentifier,
)


class FakeRetriever:
    async def aretrieve(
        self,
        *,
        query,
        tenant_id,
        top_k,
        filters,
    ):
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(
                    chunk_id="chunk-001",
                    document_id="document-001",
                    document_version_id="version-001",
                    content=(
                        "FOIR stands for Fixed Obligation "
                        "to Income Ratio."
                    ),
                    metadata={
                        "page_numbers": [5],
                        "content_type": "text",
                    },
                ),
                score=0.95,
                reranker_score=0.91,
            ),
            SimpleNamespace(
                chunk=SimpleNamespace(
                    chunk_id="chunk-002",
                    document_id="document-002",
                    document_version_id="version-001",
                    content="Other underwriting content.",
                    metadata={
                        "page_numbers": [6],
                        "content_type": "text",
                    },
                ),
                score=0.80,
                reranker_score=0.72,
            ),
        ]


@pytest.mark.asyncio
async def test_generates_candidate_source_ids():
    identifier = GoldenSourceIdentifier(
        retriever=FakeRetriever(),
        top_k=5,
    )

    results = await identifier.generate(
        [
            {
                "question": "What is FOIR?",
                "tenant_id": "contextops",
                "expected_answer": (
                    "FOIR stands for Fixed Obligation "
                    "to Income Ratio."
                ),
                "filters": None,
            }
        ]
    )

    assert len(results) == 1

    result = results[0]

    assert result["question"] == "What is FOIR?"
    assert result["tenant_id"] == "contextops"

    assert result["candidate_source_ids"] == [
        "chunk-001",
        "chunk-002",
    ]

    assert result["sources"][0]["chunk_id"] == (
        "chunk-001"
    )

    assert result["sources"][0]["page_numbers"] == [5]

    assert result["sources"][0]["reranker_score"] == (
        0.91
    )


@pytest.mark.asyncio
async def test_passes_question_tenant_top_k_and_filters():
    class RecordingRetriever:
        def __init__(self):
            self.request = None

        async def aretrieve(
            self,
            *,
            query,
            tenant_id,
            top_k,
            filters,
        ):
            self.request = {
                "query": query,
                "tenant_id": tenant_id,
                "top_k": top_k,
                "filters": filters,
            }

            return []

    retriever = RecordingRetriever()

    identifier = GoldenSourceIdentifier(
        retriever=retriever,
        top_k=20,
    )

    await identifier.generate(
        [
            {
                "question": "What is FOIR?",
                "tenant_id": "contextops",
                "expected_answer": "FOIR...",
                "filters": {
                    "content_type": "text",
                },
            }
        ]
    )

    assert retriever.request == {
        "query": "What is FOIR?",
        "tenant_id": "contextops",
        "top_k": 20,
        "filters": {
            "content_type": "text",
        },
    }