from langchain_core.documents import Document

from app.rag.retrieval.retrieval_validator import (
    RetrievalEvaluation,
    RetrievalValidator,
)


class BaselineRetrievalValidator(RetrievalValidator):
    """
    Conservative Day 18 retrieval validator.

    Current signals:
        - whether any documents were retrieved
        - whether documents contain non-empty content
        - available reranker scores

    This is intentionally a baseline.

    Day 19 will extend this with stronger:
        - reranker-score analysis
        - score separation
        - dense/BM25 agreement
        - evidence sufficiency
        - calibrated confidence thresholds
    """

    async def evaluate(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> RetrievalEvaluation:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return RetrievalEvaluation(
                sufficient=False,
                confidence="low",
                score=None,
                reason="No documents were retrieved.",
                signals={
                    "document_count": 0,
                },
            )

        non_empty_documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not non_empty_documents:
            return RetrievalEvaluation(
                sufficient=False,
                confidence="low",
                score=None,
                reason="Retrieved documents contain no usable content.",
                signals={
                    "document_count": len(documents),
                    "usable_document_count": 0,
                },
            )

        reranker_scores = [
            document.metadata.get("reranker_score")
            for document in non_empty_documents
            if document.metadata.get("reranker_score") is not None
        ]

        # --------------------------------------------------
        # Day 18 baseline decision
        # --------------------------------------------------
        #
        # We intentionally do NOT say:
        #
        #     reranker_score > X = good
        #
        # because score semantics depend on the eventual
        # reranker implementation.
        #
        # For now, usable retrieved evidence is sufficient
        # to continue.
        #

        return RetrievalEvaluation(
            sufficient=True,
            confidence="medium",
            score=None,
            reason="Usable retrieved evidence is available.",
            signals={
                "document_count": len(documents),
                "usable_document_count": len(
                    non_empty_documents
                ),
                "reranker_score_count": len(
                    reranker_scores
                ),
            },
        )