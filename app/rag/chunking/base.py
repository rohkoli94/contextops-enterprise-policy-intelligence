from abc import ABC, abstractmethod
from collections.abc import Callable

from app.domain.document_chunk import DocumentChunk
from app.domain.document_element import DocumentElement


class DocumentChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        elements: list[DocumentElement],
    ) -> list[DocumentChunk]:
        """Convert document elements into retrieval chunks."""
        raise NotImplementedError


TokenCounter = Callable[[str], int]