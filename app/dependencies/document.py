from collections.abc import Generator

from app.db.session import SessionLocal
from app.providers.storage.azure_blob import AzureBlobStorageProvider
from app.services.document_service import DocumentService


def get_document_service() -> Generator[DocumentService, None, None]:
    db = SessionLocal()

    try:
        storage_provider = AzureBlobStorageProvider()

        yield DocumentService(
            db=db,
            storage_provider=storage_provider,
        )
    finally:
        db.close()