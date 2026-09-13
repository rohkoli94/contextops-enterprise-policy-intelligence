import pytest

from app.domain.document_chunk import DocumentChunk
from app.guardrails import RegexPIIAnalyzer
from app.providers.embedding.base import (
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingProvider,
)
from app.rag.retrieval.models import RetrievedChunk
from app.rag.workflow.nodes.contextops import (
    create_contextops_node,
)
from app.tokenization.tiktoken_counter import TiktokenCounter


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for ContextOps tests.

    The provider never calls an external service.
    """

    def __init__(
        self,
        *,
        query_vector: list[float] | None = None,
        batch_vectors: list[list[float]] | None = None,
    ) -> None:
        self.query_vector = (
            query_vector
            if query_vector is not None
            else [1.0, 0.0, 0.0]
        )

        self.batch_vectors = (
            batch_vectors
            if batch_vectors is not None
            else []
        )

    def generate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            vector=list(self.query_vector),
            model="fake-model",
            provider="fake",
        )

    async def agenerate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            vector=list(self.query_vector),
            model="fake-model",
            provider="fake",
        )

    def generate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        if len(request.texts) != len(
            self.batch_vectors
        ):
            raise ValueError(
                "Unexpected number of embedding inputs."
            )

        return EmbeddingBatchResponse(
            vectors=[
                list(vector)
                for vector in self.batch_vectors
            ],
            model="fake-model",
            provider="fake",
        )

    def get_dimension(self) -> int:
        return 3


def create_test_retrieved_chunk(
    *,
    chunk_id: str,
    document_id: str,
    document_version_id: str,
    content: str,
    score: float | None,
    reranker_score: float | None,
    content_hash: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version_id=document_version_id,
            element_ids=[f"element-{chunk_id}"],
            content=content,
            chunk_index=0,
            content_hash=(
                content_hash
                if content_hash is not None
                else f"hash-{chunk_id}"
            ),
            metadata={},
        ),
        score=score if score is not None else 0.0,
        metadata={
            "document_id": document_id,
            "document_version_id": document_version_id,
            "chunk_id": chunk_id,
        },
        reranker_score=reranker_score,
    )


def create_embedding_provider_for_documents(
    vectors: list[list[float]],
) -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        query_vector=[1.0, 0.0, 0.0],
        batch_vectors=vectors,
    )


def create_pii_analyzer() -> RegexPIIAnalyzer:
    return RegexPIIAnalyzer()


@pytest.mark.asyncio
async def test_contextops_builds_context_and_citations() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            document_version_id="version-001",
            content=(
                "Managers have a three-month "
                "notice period."
            ),
            score=0.91,
            reranker_score=0.88,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            document_version_id="version-002",
            content=(
                "Employees must submit notice "
                "in writing."
            ),
            score=0.82,
            reranker_score=0.79,
        ),
    ]

    state = {
        "query": "What is the notice period?",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert (
        "Managers have a three-month notice period."
        in result["context"]
    )

    assert (
        "Employees must submit notice in writing."
        in result["context"]
    )

    assert len(result["citations"]) == 2

    assert result["citations"][0] == {
        "source": 1,
        "document_id": "doc-001",
        "document_version_id": "version-001",
        "chunk_id": "chunk-001",
        "score": 0.91,
        "reranker_score": 0.88,
    }

    assert result["citations"][1] == {
        "source": 2,
        "document_id": "doc-002",
        "document_version_id": "version-002",
        "chunk_id": "chunk-002",
        "score": 0.82,
        "reranker_score": 0.79,
    }

    assert result["context_token_count"] > 0
    assert result["context_pii_detected"] is False
    assert result["context_pii_entity_count"] == 0
    assert result["context_compressed"] is False
    assert result["context_compressed_document_count"] == 0


@pytest.mark.asyncio
async def test_contextops_uses_configured_document_limit() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=1,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-a",
            document_id="doc-a",
            document_version_id="version-a",
            content="Document A",
            score=0.90,
            reranker_score=0.80,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-b",
            document_id="doc-b",
            document_version_id="version-b",
            content="Document B",
            score=0.80,
            reranker_score=0.70,
        ),
    ]

    state = {
        "query": "test",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert len(result["citations"]) == 1
    assert result["context_token_count"] > 0


@pytest.mark.asyncio
async def test_contextops_enforces_token_budget() -> None:
    token_counter = TiktokenCounter(
        model_name="text-embedding-3-small",
    )

    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
            ]
        )
    )

    first_content = (
        "Managers are required to provide notice "
        "before leaving the organization."
    )

    second_content = (
        "Employees must submit formal written notice "
        "according to the applicable policy."
    )

    first_document = create_test_retrieved_chunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        document_version_id="version-001",
        content=first_content,
        score=0.91,
        reranker_score=2.50,
    )

    second_document = create_test_retrieved_chunk(
        chunk_id="chunk-002",
        document_id="doc-002",
        document_version_id="version-002",
        content=second_content,
        score=0.82,
        reranker_score=1.50,
    )

    first_source_block = (
        "[SOURCE 1]\n"
        "Document ID: doc-001\n"
        "Document Version ID: version-001\n"
        "Chunk ID: chunk-001\n"
        "Retrieval Score: 0.91\n"
        "Reranker Score: 2.5\n"
        f"Content:\n{first_content}"
    )

    first_source_tokens = token_counter.count(
        first_source_block
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=2,
        max_tokens=first_source_tokens,
        token_counter=token_counter,
    )

    state = {
        "query": "What is the notice policy?",
        "tenant_id": "tenant-001",
        "reranked_documents": [
            first_document,
            second_document,
        ],
    }

    result = await node(state)

    assert (
        result["context_token_count"]
        <= first_source_tokens
    )

    assert len(result["citations"]) >= 1


@pytest.mark.asyncio
async def test_contextops_skips_document_that_exceeds_budget() -> None:
    token_counter = TiktokenCounter(
        model_name="text-embedding-3-small",
    )

    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    large_content = (
        "Large policy content. "
        "This document should not fit inside "
        "the budget. "
    ) * 100

    document = create_test_retrieved_chunk(
        chunk_id="chunk-large",
        document_id="doc-large",
        document_version_id="version-large",
        content=large_content,
        score=0.90,
        reranker_score=2.00,
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
        max_tokens=10,
        token_counter=token_counter,
    )

    state = {
        "query": "test",
        "tenant_id": "tenant-001",
        "reranked_documents": [document],
    }

    result = await node(state)

    assert (
        result["context"]
        == "No relevant policy documents were found."
    )

    assert result["citations"] == []
    assert result["context_token_count"] == 0


@pytest.mark.asyncio
async def test_contextops_deduplicates_identical_content_hashes() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            document_version_id="version-001",
            content=(
                "Managers must provide three months "
                "notice."
            ),
            score=0.95,
            reranker_score=2.90,
            content_hash="same-content-hash",
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            document_version_id="version-002",
            content=(
                "Managers must provide three months "
                "notice."
            ),
            score=0.80,
            reranker_score=1.90,
            content_hash="same-content-hash",
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-003",
            document_id="doc-003",
            document_version_id="version-003",
            content=(
                "Notice must be submitted in writing."
            ),
            score=0.75,
            reranker_score=1.50,
            content_hash="unique-content-hash",
        ),
    ]

    state = {
        "query": "notice policy",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert len(result["citations"]) == 2
    assert result["citations"][0]["chunk_id"] == "chunk-001"
    assert result["citations"][1]["chunk_id"] == "chunk-003"


@pytest.mark.asyncio
async def test_contextops_deduplicates_normalized_content() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-a",
            document_id="doc-a",
            document_version_id="version-a",
            content=(
                "Managers must provide three months "
                "notice."
            ),
            score=0.95,
            reranker_score=2.90,
            content_hash="",
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-b",
            document_id="doc-b",
            document_version_id="version-b",
            content=(
                "Managers   must provide three months "
                "notice."
            ),
            score=0.80,
            reranker_score=1.90,
            content_hash="",
        ),
    ]

    state = {
        "query": "notice policy",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-a"


@pytest.mark.asyncio
async def test_contextops_preserves_highest_ranked_duplicate() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="high-ranked",
            document_id="doc-high",
            document_version_id="version-high",
            content="Same policy content.",
            score=0.99,
            reranker_score=3.00,
            content_hash="duplicate",
        ),
        create_test_retrieved_chunk(
            chunk_id="low-ranked",
            document_id="doc-low",
            document_version_id="version-low",
            content="Same policy content.",
            score=0.70,
            reranker_score=1.00,
            content_hash="duplicate",
        ),
    ]

    state = {
        "query": "policy",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert len(result["citations"]) == 1
    assert (
        result["citations"][0]["chunk_id"]
        == "high-ranked"
    )


@pytest.mark.asyncio
async def test_contextops_mmr_selects_diverse_document() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )

    embedding_provider.query_vector = [
        1.0,
        0.0,
        0.0,
    ]

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=2,
        mmr_lambda=0.5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            document_version_id="version-001",
            content="Primary policy about leave approval.",
            score=0.99,
            reranker_score=3.00,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            document_version_id="version-002",
            content="Another version of the leave approval policy.",
            score=0.98,
            reranker_score=2.95,
        ),
        create_test_retrieved_chunk(
            chunk_id="chunk-003",
            document_id="doc-003",
            document_version_id="version-003",
            content="Policy about employee remote work.",
            score=0.97,
            reranker_score=2.90,
        ),
    ]

    state = {
        "query": "employee policy",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    selected_chunk_ids = [
        citation["chunk_id"]
        for citation in result["citations"]
    ]

    assert "chunk-001" in selected_chunk_ids
    assert "chunk-002" not in selected_chunk_ids
    assert "chunk-003" in selected_chunk_ids


@pytest.mark.asyncio
async def test_contextops_redacts_email_from_retrieved_context() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-pii",
            document_id="doc-pii",
            document_version_id="version-pii",
            content=(
                "Contact employee@example.com "
                "for policy clarification."
            ),
            score=0.95,
            reranker_score=2.90,
        ),
    ]

    state = {
        "query": "policy clarification",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert "employee@example.com" not in result["context"]
    assert "[REDACTED_EMAIL]" in result["context"]
    assert result["context_pii_detected"] is True
    assert result["context_pii_entity_count"] == 1


@pytest.mark.asyncio
async def test_contextops_redacts_multiple_pii_entities() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-pii",
            document_id="doc-pii",
            document_version_id="version-pii",
            content=(
                "Employee email is employee@example.com "
                "and phone is 9876543210."
            ),
            score=0.95,
            reranker_score=2.90,
        ),
    ]

    state = {
        "query": "employee contact policy",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert "employee@example.com" not in result["context"]
    assert "9876543210" not in result["context"]

    assert "[REDACTED_EMAIL]" in result["context"]
    assert "[REDACTED_PHONE]" in result["context"]

    assert result["context_pii_detected"] is True
    assert result["context_pii_entity_count"] == 2


@pytest.mark.asyncio
async def test_contextops_redacts_aadhaar_like_number() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=5,
    )

    documents = [
        create_test_retrieved_chunk(
            chunk_id="chunk-aadhaar",
            document_id="doc-aadhaar",
            document_version_id="version-aadhaar",
            content=(
                "Employee identification number: "
                "1234 5678 9012."
            ),
            score=0.95,
            reranker_score=2.90,
        ),
    ]

    state = {
        "query": "employee identification policy",
        "tenant_id": "tenant-001",
        "reranked_documents": documents,
    }

    result = await node(state)

    assert "1234 5678 9012" not in result["context"]
    assert "[REDACTED_AADHAAR]" in result["context"]
    assert result["context_pii_detected"] is True
    assert result["context_pii_entity_count"] == 1


@pytest.mark.asyncio
async def test_contextops_compresses_long_context() -> None:
    token_counter = TiktokenCounter(
        model_name="text-embedding-3-small",
    )

    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    content = (
        "Managers have a three-month notice period. "
        "This sentence contains unrelated administrative "
        "information about office facilities. "
        "Employees must submit notice in writing. "
        "This sentence contains unrelated payroll details "
        "that are not relevant to the question."
    )

    document = create_test_retrieved_chunk(
        chunk_id="compressed-001",
        document_id="doc-compressed",
        document_version_id="version-compressed",
        content=content,
        score=0.95,
        reranker_score=2.90,
    )

    source_prefix = (
        "[SOURCE 1]\n"
        "Document ID: doc-compressed\n"
        "Document Version ID: version-compressed\n"
        "Chunk ID: compressed-001\n"
        "Retrieval Score: 0.95\n"
        "Reranker Score: 2.9\n"
        "Content:\n"
    )

    prefix_tokens = token_counter.count(
        source_prefix
    )

    first_sentence = (
        "Managers have a three-month notice period."
    )

    second_sentence = (
        "Employees must submit notice in writing."
    )

    target_content_tokens = token_counter.count(
        f"{first_sentence} {second_sentence}"
    )

    max_tokens = (
        prefix_tokens
        + target_content_tokens
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=1,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )

    state = {
        "query": "What is the notice period?",
        "tenant_id": "tenant-001",
        "reranked_documents": [document],
    }

    result = await node(state)

    assert result["context_compressed"] is True
    assert (
        result["context_compressed_document_count"]
        == 1
    )

    assert (
        "Managers have a three-month notice period."
        in result["context"]
    )

    assert (
        "Employees must submit notice in writing."
        in result["context"]
    )

    assert (
        "unrelated administrative information"
        not in result["context"]
    )

    assert (
        result["context_token_count"]
        <= max_tokens
    )


@pytest.mark.asyncio
async def test_contextops_keeps_short_content_uncompressed() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    content = (
        "Managers have a three-month notice period."
    )

    document = create_test_retrieved_chunk(
        chunk_id="short-001",
        document_id="doc-short",
        document_version_id="version-short",
        content=content,
        score=0.95,
        reranker_score=2.90,
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=1,
        max_tokens=1000,
    )

    state = {
        "query": "notice period",
        "tenant_id": "tenant-001",
        "reranked_documents": [document],
    }

    result = await node(state)

    assert result["context_compressed"] is False
    assert (
        result["context_compressed_document_count"]
        == 0
    )

    assert content in result["context"]


@pytest.mark.asyncio
async def test_contextops_compression_preserves_source_metadata() -> None:
    embedding_provider = (
        create_embedding_provider_for_documents(
            [
                [1.0, 0.0, 0.0],
            ]
        )
    )

    document = create_test_retrieved_chunk(
        chunk_id="metadata-001",
        document_id="doc-metadata",
        document_version_id="version-metadata",
        content=(
            "Managers have a three-month notice period. "
            "Unrelated administrative content follows."
        ),
        score=0.95,
        reranker_score=2.90,
    )

    node = create_contextops_node(
        embedding_provider=embedding_provider,
        pii_analyzer=create_pii_analyzer(),
        max_documents=1,
        max_tokens=80,
    )

    state = {
        "query": "notice period",
        "tenant_id": "tenant-001",
        "reranked_documents": [document],
    }

    result = await node(state)

    assert "[SOURCE 1]" in result["context"]
    assert "Document ID: doc-metadata" in result["context"]
    assert (
        "Document Version ID: version-metadata"
        in result["context"]
    )
    assert "Chunk ID: metadata-001" in result["context"]

    assert len(result["citations"]) == 1

    assert (
        result["citations"][0]["document_id"]
        == "doc-metadata"
    )


def test_contextops_rejects_invalid_document_limit() -> None:
    embedding_provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError):
        create_contextops_node(
            embedding_provider=embedding_provider,
            pii_analyzer=create_pii_analyzer(),
            max_documents=0,
        )


def test_contextops_rejects_invalid_token_limit() -> None:
    embedding_provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError):
        create_contextops_node(
            embedding_provider=embedding_provider,
            pii_analyzer=create_pii_analyzer(),
            max_tokens=0,
        )


def test_contextops_rejects_invalid_mmr_lambda() -> None:
    embedding_provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError):
        create_contextops_node(
            embedding_provider=embedding_provider,
            pii_analyzer=create_pii_analyzer(),
            mmr_lambda=1.5,
        )


def test_contextops_rejects_negative_mmr_lambda() -> None:
    embedding_provider = FakeEmbeddingProvider()

    with pytest.raises(ValueError):
        create_contextops_node(
            embedding_provider=embedding_provider,
            pii_analyzer=create_pii_analyzer(),
            mmr_lambda=-0.1,
        )