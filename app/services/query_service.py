from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config.settings import settings
from app.providers.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from app.rag.retrieval.langchain_retriever import (
    LangChainRetrieverAdapter,
)

from app.api.v1.query.schemas.query_filter import (
    QueryFilter,
)

class QueryService:
    """
    Application service responsible for answering
    enterprise policy questions.

    Query flow:

        User Question
            ↓
        QueryService
            ↓
        LangChain Retriever
            ↓
        Hybrid Retrieval
            ↓
        Retrieved Documents
            ↓
        Context Construction
            ↓
        LLM
            ↓
        Grounded Answer
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        hybrid_retriever: BaseRetriever,
    ) -> None:
        self.llm_provider = llm_provider
        self.hybrid_retriever = hybrid_retriever

    async def ask(
        self,
        question: str,
        tenant_id: str,
        filters: QueryFilter | None = None,
    ) -> LLMResponse:
        """
        Answer a user question using tenant-aware hybrid
        retrieval and the configured LLM provider.
        """

        retrieval_filters = (
            filters.model_dump(exclude_none=True)
            if filters
            else None
        )
        
        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        # --------------------------------------------------
        # STEP 1 — RETRIEVAL CONFIGURATION
        # --------------------------------------------------

        # top_k comes from application configuration.
        #
        # It is intentionally not hardcoded here.
        #
        # Example:
        #
        # RETRIEVAL_TOP_K=10

        retriever = LangChainRetrieverAdapter(
            hybrid_retriever=self.hybrid_retriever,
            tenant_id=tenant_id,
            top_k=settings.retrieval_top_k,
            filters=retrieval_filters,
        )

        # --------------------------------------------------
        # STEP 2 — ASYNC HYBRID RETRIEVAL
        # --------------------------------------------------

        documents = await retriever.ainvoke(
            question
        )

        # --------------------------------------------------
        # STEP 3 — BUILD LLM CONTEXT
        # --------------------------------------------------

        context = self._build_context(
            documents
        )

        # --------------------------------------------------
        # STEP 4 — BUILD LLM REQUEST
        # --------------------------------------------------

        request = LLMRequest(
            system_prompt=(
                "You are an enterprise policy intelligence "
                "assistant. Answer the user's question using "
                "only the provided policy context. "
                "If the context does not contain enough "
                "information, clearly state that the answer "
                "cannot be determined from the available "
                "policy documents. Do not hallucinate."
            ),
            user_prompt=question,
            context=context,
        )

        # --------------------------------------------------
        # STEP 5 — GENERATE ANSWER
        # --------------------------------------------------

        return await self.llm_provider.agenerate(
            request
        )

    # ========================================================
    # CONTEXT CONSTRUCTION
    # ========================================================

    @staticmethod
    def _build_context(
        documents: list[Document],
    ) -> str:
        """
        Convert LangChain Documents into a structured,
        LLM-ready evidence context.
        """

        if not documents:
            return (
                "No relevant policy documents were found."
            )

        context_parts: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            context_parts.append(
                (
                    f"[SOURCE {index}]\n"
                    f"Document ID: "
                    f"{metadata.get('document_id', '')}\n"
                    f"Version ID: "
                    f"{metadata.get('document_version_id', '')}\n"
                    f"Chunk ID: "
                    f"{metadata.get('chunk_id', '')}\n"
                    f"Retrieval Score: "
                    f"{metadata.get('score', '')}\n"
                    f"Content:\n"
                    f"{document.page_content}"
                )
            )

        return "\n\n".join(
            context_parts
        )