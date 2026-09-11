from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.reranker import Reranker


class PassThroughReranker(Reranker):
    """
    Baseline reranker implementation.

    Preserves the original retrieval ordering and retrieval scores
    while explicitly marking that no real reranking model was applied.
    """

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        reranked_candidates: list[RetrievedChunk] = []

        for rank, candidate in enumerate(candidates, start=1):
            metadata = dict(candidate.metadata)

            metadata["reranker_applied"] = False
            metadata["reranker_rank"] = rank
            metadata["reranker_score"] = None

            reranked_candidates.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    score=candidate.score,
                    metadata=metadata,
                    reranker_score=None,
                )
            )

        return reranked_candidates