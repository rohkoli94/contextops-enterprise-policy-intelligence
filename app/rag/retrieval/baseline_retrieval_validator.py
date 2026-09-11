from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_validator import (
    RetrievalEvaluation,
    RetrievalValidator,
)


class BaselineRetrievalValidator(RetrievalValidator):
    """
    Day 19 retrieval confidence validator.

    Confidence levels:

        none
            No usable evidence was retrieved.

        weak
            Some usable evidence exists, but the evidence is
            not strong enough to confidently answer the query.

        sufficient
            Usable evidence exists and appears relevant enough
            to continue to ContextOps / LLM generation.

        strong
            Strong reranked evidence with clear score separation
            is available.

    Reranker scores are treated only as ranking signals.
    They are NOT interpreted as probabilities.

    The thresholds are intentionally baseline heuristics and
    should be calibrated later using evaluation data.
    """

    MIN_USABLE_DOCUMENTS = 1
    STRONG_SCORE_GAP = 1.0

    async def evaluate(
        self,
        *,
        query: str,
        documents: list[RetrievedChunk],
    ) -> RetrievalEvaluation:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # ------------------------------------------------------
        # NONE
        # ------------------------------------------------------
        if not documents:
            return RetrievalEvaluation(
                sufficient=False,
                confidence="none",
                score=None,
                reason="No documents were retrieved.",
                signals={
                    "document_count": 0,
                    "usable_document_count": 0,
                    "reranker_score_count": 0,
                    "top_reranker_score": None,
                    "second_reranker_score": None,
                    "reranker_score_gap": None,
                },
            )

        # ------------------------------------------------------
        # Remove documents with empty content
        # ------------------------------------------------------
        usable_documents = [
            document
            for document in documents
            if document.chunk.content
            and document.chunk.content.strip()
        ]

        # ------------------------------------------------------
        # NONE
        # ------------------------------------------------------
        if len(usable_documents) < self.MIN_USABLE_DOCUMENTS:
            return RetrievalEvaluation(
                sufficient=False,
                confidence="none",
                score=None,
                reason=(
                    "Retrieved documents contain no usable content."
                ),
                signals={
                    "document_count": len(documents),
                    "usable_document_count": 0,
                    "reranker_score_count": 0,
                    "top_reranker_score": None,
                    "second_reranker_score": None,
                    "reranker_score_gap": None,
                },
            )

        # ------------------------------------------------------
        # Collect reranker scores
        # ------------------------------------------------------
        reranker_scores = [
            float(document.reranker_score)
            for document in usable_documents
            if document.reranker_score is not None
        ]

        reranker_scores.sort(reverse=True)

        top_score = (
            reranker_scores[0]
            if reranker_scores
            else None
        )

        second_score = (
            reranker_scores[1]
            if len(reranker_scores) > 1
            else None
        )

        score_gap = (
            top_score - second_score
            if top_score is not None
            and second_score is not None
            else None
        )

        signals: dict[str, object] = {
            "document_count": len(documents),
            "usable_document_count": len(usable_documents),
            "reranker_score_count": len(reranker_scores),
            "top_reranker_score": top_score,
            "second_reranker_score": second_score,
            "reranker_score_gap": score_gap,
        }

        # ------------------------------------------------------
        # SUFFICIENT
        # ------------------------------------------------------
        #
        # We have usable evidence, but no reranker score.
        #
        # This is enough to continue the workflow, but we do
        # not claim strong confidence.
        #
        if top_score is None:
            return RetrievalEvaluation(
                sufficient=True,
                confidence="sufficient",
                score=None,
                reason=(
                    "Usable retrieved evidence is available, "
                    "but reranker scores are unavailable."
                ),
                signals=signals,
            )

        # ------------------------------------------------------
        # STRONG
        # ------------------------------------------------------
        #
        # A clear separation between the top result and the
        # second result indicates stronger ranking confidence.
        #
        if (
            len(reranker_scores) >= 2
            and score_gap is not None
            and score_gap > self.STRONG_SCORE_GAP
        ):
            return RetrievalEvaluation(
                sufficient=True,
                confidence="strong",
                score=top_score,
                reason=(
                    "Strong reranked evidence with clear "
                    "score separation is available."
                ),
                signals=signals,
            )

        # ------------------------------------------------------
        # SUFFICIENT
        # ------------------------------------------------------
        #
        # Usable reranked evidence exists, but there is not
        # enough separation to classify it as strong.
        #
        return RetrievalEvaluation(
            sufficient=True,
            confidence="sufficient",
            score=top_score,
            reason=(
                "Usable reranked evidence is available, "
                "but confidence is not strongly separated."
            ),
            signals=signals,
        )