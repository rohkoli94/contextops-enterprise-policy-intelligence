from typing import Any

import pytest
from langchain_core.documents import Document

from app.guardrails.authorization import AuthorizationGuard
from app.guardrails.pii import RegexPIIAnalyzer
from app.guardrails.prompt_injection import PromptInjectionGuard
from app.guardrails.tenant_isolation import TenantIsolationGuard
from app.rag.retrieval.grounding_validator import (
    GroundingEvaluation,
)
from app.rag.retrieval.retrieval_validator import (
    RetrievalEvaluation,
)
from app.rag.workflow.graph import create_query_graph


# ============================================================
# TEST LLM
# ============================================================


class FakeLLMProvider:
    async def agenerate(
        self,
        request: Any,
    ):
        class Response:
            content = (
                "The notice period is three months. "
                "[SOURCE 1]"
            )

        return Response()


# ============================================================
# TEST RETRIEVER
# ============================================================


class FakeHybridRetriever:
    async def aretrieve(
        self,
        *,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ):
        return [
            type(
                "Retrieved",
                (),
                {
                    "chunk": type(
                        "Chunk",
                        (),
                        {
                            "content": (
                                "The notice period is "
                                "three months."
                            ),
                            "chunk_id": "chunk-001",
                            "document_id": "doc-001",
                            "document_version_id": (
                                "version-001"
                            ),
                        },
                    )(),
                    "score": 0.92,
                },
            )()
        ]


# ============================================================
# TEST CONVERSATION MEMORY
# ============================================================


class FakeConversationMemory:
    async def get_context(
        self,
        conversation_id: str,
        tenant_id: str,
        *,
        recent_message_limit: int,
    ):
        class Message:
            def __init__(self) -> None:
                self.message_id = "message-001"
                self.conversation_id = (
                    conversation_id
                )
                self.role = type(
                    "Role",
                    (),
                    {
                        "value": "user",
                    },
                )()
                self.content = (
                    "What is the notice period?"
                )
                self.created_at = type(
                    "Date",
                    (),
                    {
                        "isoformat": (
                            lambda self:
                            "2026-09-04T12:00:00+00:00"
                        ),
                    },
                )()

        class Context:
            recent_messages = [
                Message()
            ]
            summary = (
                "User is asking about notice "
                "period policy."
            )

        return Context()


# ============================================================
# TEST QUERY REWRITER
# ============================================================


class FakeQueryRewriter:
    async def rewrite(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        recent_messages: list[dict[str, object]],
    ) -> str:
        return (
            "What is the notice period policy "
            "for managers?"
        )


# ============================================================
# TEST CACHE
# ============================================================


class FakeCacheProvider:
    async def get(
        self,
        key: str,
    ):
        return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        return None

    async def delete(
        self,
        key: str,
    ) -> None:
        return None


# ============================================================
# TEST RERANKER
# ============================================================


class FakeReranker:
    async def rerank(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        return documents


# ============================================================
# TEST RETRIEVAL VALIDATOR
# ============================================================


class FakeRetrievalValidator:
    async def evaluate(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> RetrievalEvaluation:
        return RetrievalEvaluation(
            sufficient=True,
            confidence="high",
            score=0.95,
            reason="Strong evidence found.",
            signals={
                "test": True,
            },
        )


# ============================================================
# TEST GROUNDING VALIDATOR
# ============================================================


class FakeGroundingValidator:
    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        documents: list[Document],
    ) -> GroundingEvaluation:
        return GroundingEvaluation(
            grounded=True,
            confidence="high",
            reason="Answer is supported by evidence.",
            supported_sources=[1],
        )


# ============================================================
# END-TO-END GRAPH TEST
# ============================================================


@pytest.mark.asyncio
async def test_query_graph_executes_end_to_end() -> None:
    graph = create_query_graph(
        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        llm_provider=FakeLLMProvider(),

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        hybrid_retriever=FakeHybridRetriever(),

        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

        conversation_memory=(
            FakeConversationMemory()
        ),

        query_rewriter=(
            FakeQueryRewriter()
        ),

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        cache_provider=(
            FakeCacheProvider()
        ),

        # ----------------------------------------------------
        # REAL DAY 18 GUARDRAILS
        # ----------------------------------------------------

        authorization_guard=(
            AuthorizationGuard()
        ),

        tenant_isolation_guard=(
            TenantIsolationGuard()
        ),

        prompt_injection_guard=(
            PromptInjectionGuard()
        ),

        pii_analyzer=(
            RegexPIIAnalyzer()
        ),

        # ----------------------------------------------------
        # Retrieval pipeline
        # ----------------------------------------------------

        reranker=(
            FakeReranker()
        ),

        retrieval_validator=(
            FakeRetrievalValidator()
        ),

        grounding_validator=(
            FakeGroundingValidator()
        ),
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    result = await graph.ainvoke(
        {
            "query": "What about managers?",
            "tenant_id": "tenant-001",
            "conversation_id": None,
            "filters": None,
            "retry_count": 0,
        }
    )

    # ========================================================
    # ORIGINAL QUERY
    # ========================================================

    assert (
        result["query"]
        == "What about managers?"
    )

    # ========================================================
    # QUERY CONTEXTUALIZATION
    # ========================================================

    assert (
        result["contextualized_query"]
        == (
            "What is the notice period policy "
            "for managers?"
        )
    )

    assert (
        result["query_rewritten"]
        is True
    )

    assert (
        result["query_rewrite_fallback"]
        is False
    )

    # ========================================================
    # SECURITY
    # ========================================================

    assert (
        result["input_valid"]
        is True
    )

    assert (
        result["authorization_valid"]
        is True
    )

    assert (
        result["tenant_valid"]
        is True
    )

    assert (
        result["prompt_injection_detected"]
        is False
    )

    assert (
        result["input_pii_detected"]
        is False
    )

    # ========================================================
    # CACHE
    # ========================================================

    assert (
        result["cache_hit"]
        is False
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    assert (
        result["retrieved_documents"]
    )

    # ========================================================
    # RERANKING
    # ========================================================

    assert (
        result["reranked_documents"]
    )

    # ========================================================
    # RETRIEVAL VALIDATION
    # ========================================================

    assert (
        result["retrieval_sufficient"]
        is True
    )

    assert (
        result["retrieval_confidence"]
        == "high"
    )

    assert (
        result["retrieval_score"]
        == 0.95
    )

    # ========================================================
    # CONTEXTOPS
    # ========================================================

    assert (
        "[SOURCE 1]"
        in result["context"]
    )

    assert (
        result["citations"]
    )

    # ========================================================
    # LLM
    # ========================================================

    assert (
        result["answer"]
        == (
            "The notice period is three months. "
            "[SOURCE 1]"
        )
    )

    # ========================================================
    # GROUNDING
    # ========================================================

    assert (
        result["grounding_status"]
        == "grounded"
    )

    assert (
        result["grounding_supported_sources"]
        == [1]
    )

    # ========================================================
    # FINAL RESPONSE METADATA
    # ========================================================

    assert (
        result["final_response_metadata"]
    )

    assert (
        result["final_response_metadata"][
            "cache_hit"
        ]
        is False
    )