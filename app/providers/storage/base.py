from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageProvider(ABC):

    @abstractmethod
    def upload(
        self,
        path: str,
        stream: BinaryIO,
        content_type: str,
    ) -> str:
        """Upload content from a stream and return its storage path."""
        raise NotImplementedError


    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete content from storage."""
        raise NotImplementedError