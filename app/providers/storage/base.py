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