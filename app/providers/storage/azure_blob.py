from typing import BinaryIO

from azure.storage.blob import BlobServiceClient, ContentSettings

from app.config.settings import settings
from app.providers.storage.base import StorageProvider


class AzureBlobStorageProvider(StorageProvider):
    def __init__(self) -> None:
        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )

        self.container_client = (
            self.blob_service_client.get_container_client(
                settings.azure_storage_container_name
            )
        )

    def upload(
        self,
        path: str,
        stream: BinaryIO,
        content_type: str,
    ) -> str:
        blob_client = self.container_client.get_blob_client(path)

        blob_client.upload_blob(
            data=stream,
            overwrite=False,
            content_settings=ContentSettings(
                content_type=content_type
            ),
        )

        return path

    def delete(
            self,
            path: str
        ) -> None:
            blob_client = self.container_client.get_blob_client(path)
    
            blob_client.delete_blob()
    