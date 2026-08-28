from abc import ABC, abstractmethod

from app.rag.retrieval.models import RetrievedChunk


class RetrievalProvider(ABC):
    """
    Abstraction for retrieving relevant document chunks.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks.
        """
        raise NotImplementedError