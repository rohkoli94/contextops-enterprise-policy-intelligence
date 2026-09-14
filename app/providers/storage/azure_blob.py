import tempfile
from typing import BinaryIO

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
)
from azure.storage.blob.aio import (
    BlobServiceClient as AsyncBlobServiceClient,
)

from app.config.settings import settings
from app.providers.storage.base import StorageProvider


class AzureBlobStorageProvider(StorageProvider):
    """
    Azure Blob Storage implementation.

    Supports both synchronous and asynchronous operations.

    Async operations use the Azure asynchronous SDK directly.
    """

    def __init__(self) -> None:
        # ========================================================
        # SYNCHRONOUS CLIENT
        # ========================================================

        self.blob_service_client = (
            BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string,
            )
        )

        self.container_client = (
            self.blob_service_client.get_container_client(
                settings.azure_storage_container_name,
            )
        )

        # ========================================================
        # ASYNCHRONOUS CLIENT
        # ========================================================

        self.async_blob_service_client = (
            AsyncBlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string,
            )
        )

        self.async_container_client = (
            self.async_blob_service_client.get_container_client(
                settings.azure_storage_container_name,
            )
        )

    # ============================================================
    # UPLOAD
    # ============================================================

    def upload(
        self,
        path: str,
        stream: BinaryIO,
        content_type: str,
    ) -> str:
        blob_client = (
            self.container_client.get_blob_client(path)
        )

        blob_client.upload_blob(
            data=stream,
            overwrite=False,
            content_settings=ContentSettings(
                content_type=content_type,
            ),
        )

        return path

    async def aupload(
        self,
        path: str,
        stream: BinaryIO,
        content_type: str,
    ) -> str:
        blob_client = (
            self.async_container_client.get_blob_client(path)
        )

        await blob_client.upload_blob(
            data=stream,
            overwrite=False,
            content_settings=ContentSettings(
                content_type=content_type,
            ),
        )

        return path

    # ============================================================
    # DOWNLOAD
    # ============================================================

    def download(
        self,
        path: str,
    ) -> BinaryIO:
        """
        Download a blob into a temporary spooled file.

        Small files remain in memory while larger files spill to
        disk, avoiding an unconditional full-memory document copy.
        """

        blob_client = (
            self.container_client.get_blob_client(path)
        )

        downloader = blob_client.download_blob()

        stream = tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024,
            mode="w+b",
        )

        for chunk in downloader.chunks():
            stream.write(chunk)

        stream.seek(0)

        return stream

    async def adownload(
        self,
        path: str,
    ) -> BinaryIO:
        """
        Download a blob asynchronously using the Azure async SDK.

        The returned stream is positioned at the beginning.
        """

        blob_client = (
            self.async_container_client.get_blob_client(path)
        )

        downloader = await blob_client.download_blob()

        stream = tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024,
            mode="w+b",
        )

        async for chunk in downloader.chunks():
            stream.write(chunk)

        stream.seek(0)

        return stream

    # ============================================================
    # DELETE
    # ============================================================

    def delete(
        self,
        path: str,
    ) -> None:
        blob_client = (
            self.container_client.get_blob_client(path)
        )

        blob_client.delete_blob()

    async def adelete(
        self,
        path: str,
    ) -> None:
        blob_client = (
            self.async_container_client.get_blob_client(path)
        )

        await blob_client.delete_blob()

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def close(self) -> None:
        self.blob_service_client.close()

    async def aclose(self) -> None:
        await self.async_blob_service_client.close()