from abc import ABC, abstractmethod
from typing import BinaryIO

from app.domain.document import Document
from app.domain.document_element import DocumentElement


class DocumentExtractor(ABC):

    @abstractmethod
    def extract(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str
    ) -> list[DocumentElement]:
        raise NotImplementedError