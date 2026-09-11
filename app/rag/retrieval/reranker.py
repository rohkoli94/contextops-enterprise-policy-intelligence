from abc import ABC, abstractmethod

from app.rag.retrieval.models import RetrievedChunk


class Reranker(ABC):
    """
    Abstraction for reranking retrieved candidate chunks.

    Retrieval is responsible for recall.

    Reranking is responsible for improving precision by
    reordering the retrieved candidate set according to
    query-document relevance.

    The original retrieval score is preserved separately from
    the reranker score.
    """

    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Rerank the supplied retrieval candidates.

        Implementations must:

        - preserve the original RetrievedChunk objects/content
        - preserve the original retrieval score
        - preserve tenant/filter metadata
        - attach the reranker score separately
        - return candidates ordered by reranker relevance
        """
        raise NotImplementedError