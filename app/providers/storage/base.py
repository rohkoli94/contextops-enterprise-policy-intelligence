import asyncio
from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageProvider(ABC):
    """
    Abstraction for document object storage.

    Synchronous methods remain available for backward compatibility.

    Async methods are the preferred interface for the FastAPI
    ingestion path.

    By default, async methods delegate synchronous implementations
    to a worker thread so blocking SDK calls do not block the
    application event loop.

    Providers with native asynchronous SDKs should override the
    async methods.
    """

    # ============================================================
    # UPLOAD
    # ============================================================

    @abstractmethod
    def upload(
        self,
        path: str,
        stream: BinaryIO,
        content_type: str,
    ) -> str:
        """Upload content synchronously."""
        raise NotImplementedError

    async def aupload(
        self,
        path: str,
        stream: BinaryIO,
        content_type: str,
    ) -> str:
        """Upload content asynchronously."""
        return await asyncio.to_thread(
            self.upload,
            path,
            stream,
            content_type,
        )

    # ============================================================
    # DOWNLOAD
    # ============================================================

    @abstractmethod
    def download(
        self,
        path: str,
    ) -> BinaryIO:
        """
        Download content synchronously.

        Returns:
            A readable binary stream positioned at the beginning.
        """
        raise NotImplementedError

    async def adownload(
        self,
        path: str,
    ) -> BinaryIO:
        """Download content asynchronously."""
        return await asyncio.to_thread(
            self.download,
            path,
        )

    # ============================================================
    # DELETE
    # ============================================================

    @abstractmethod
    def delete(
        self,
        path: str,
    ) -> None:
        """Delete content synchronously."""
        raise NotImplementedError

    async def adelete(
        self,
        path: str,
    ) -> None:
        """Delete content asynchronously."""
        await asyncio.to_thread(
            self.delete,
            path,
        )