from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Reranker(ABC):
    """
    Abstraction for reranking retrieved candidate documents.

    Retrieval is responsible for recall.

    Reranking is responsible for improving precision by
    reordering the retrieved candidate set according to
    query-document relevance.
    """

    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """
        Rerank candidate documents for the supplied query.

        Implementations must preserve the Document objects and
        may enrich metadata with reranking information.
        """
        raise NotImplementedError