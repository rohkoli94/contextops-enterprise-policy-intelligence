from langchain_core.documents import Document

from app.rag.retrieval.reranker import Reranker


class PassThroughReranker(Reranker):
    """
    Day 18 baseline reranker.

    The component preserves retrieval ordering and records
    that no reranking model has been applied yet.

    Day 19 will replace this with a real reranker implementation.
    """

    async def rerank(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        reranked_documents: list[Document] = []

        for rank, document in enumerate(
            documents,
            start=1,
        ):
            metadata = dict(document.metadata)

            metadata["reranker_applied"] = False
            metadata["reranker_rank"] = rank
            metadata["reranker_score"] = None

            reranked_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return reranked_documents