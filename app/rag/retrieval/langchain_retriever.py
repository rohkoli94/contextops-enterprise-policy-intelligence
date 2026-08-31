from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.rag.retrieval.hybrid import HybridRetriever


class LangChainRetrieverAdapter(BaseRetriever):
    """
    Adapts the ContextOps HybridRetriever to LangChain's
    standard retriever interface.

    ContextOps owns the actual retrieval logic:

        - dense retrieval
        - sparse BM25 retrieval
        - tenant routing
        - metadata filtering
        - Qdrant hybrid search
        - RRF fusion

    LangChain provides the standard retriever interface
    for integration with LangChain and LangGraph.
    """

    # --------------------------------------------------------
    # Pydantic configuration
    # --------------------------------------------------------

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # --------------------------------------------------------
    # Dependencies
    # --------------------------------------------------------

    hybrid_retriever: HybridRetriever

    # --------------------------------------------------------
    # Request-level retrieval context
    # --------------------------------------------------------

    tenant_id: str

    top_k: int

    filters: dict[str, Any] | None = None

    # ========================================================
    # ASYNC RETRIEVAL
    # ========================================================

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """
        Execute asynchronous ContextOps hybrid retrieval
        and convert the results into LangChain Documents.

        LangChain's ainvoke() calls this method.
        """

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # ----------------------------------------------------
        # CONTEXTOPS HYBRID RETRIEVAL
        # ----------------------------------------------------

        results = await self.hybrid_retriever.aretrieve(
            query=query,
            tenant_id=self.tenant_id,
            top_k=self.top_k,
            filters=self.filters,
        )

        # ----------------------------------------------------
        # LANGCHAIN DOCUMENTS
        # ----------------------------------------------------

        return self._to_langchain_documents(
            results
        )

    # ========================================================
    # RESULT CONVERSION
    # ========================================================

    @staticmethod
    def _to_langchain_documents(
        results: list[Any],
    ) -> list[Document]:
        """
        Convert ContextOps RetrievedChunk objects into
        LangChain Document objects.
        """

        documents: list[Document] = []

        for result in results:
            chunk = result.chunk

            metadata = {
                **result.metadata,

                # Retrieval information
                "score": float(result.score),

                # Chunk identity
                "chunk_id": str(
                    chunk.chunk_id
                ),

                # Document identity
                "document_id": str(
                    chunk.document_id
                ),

                "document_version_id": str(
                    chunk.document_version_id
                ),
            }

            documents.append(
                Document(
                    page_content=chunk.content,
                    metadata=metadata,
                )
            )

        return documents