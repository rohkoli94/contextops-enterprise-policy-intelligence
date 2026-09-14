import uuid
from datetime import datetime, timezone
from io import BytesIO

import pytest

from app.domain.document import Document
from app.domain.document_chunk import DocumentChunk
from app.domain.document_element import (
    ContentType,
    DocumentElement,
)
from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)
from app.domain.sparse_embedding import SparseEmbedding
from app.providers.embedding.base import (
    EmbeddingBatchResponse,
)
from app.services.document_ingestion_service import (
    DocumentIngestionService,
)


class FakeStorageProvider:
    def __init__(self) -> None:
        self.downloaded_paths: list[str] = []

    async def adownload(
        self,
        path: str,
    ):
        self.downloaded_paths.append(path)

        return BytesIO(
            b"fake document content"
        )


class FakeExtractor:
    def __init__(self) -> None:
        self.received_stream = None
        self.received_file_name = None
        self.received_version_id = None

    async def aextract(
        self,
        *,
        document,
        stream,
        file_name,
        document_version_id,
    ):
        self.received_stream = stream
        self.received_file_name = file_name
        self.received_version_id = (
            document_version_id
        )

        return [
            DocumentElement(
                element_id="element-001",
                document_id=document.document_id,
                page_number=1,
                content_type=ContentType.TEXT,
                content="Employees can work remotely.",
                content_hash="element-hash-001",
            ),
            DocumentElement(
                element_id="element-002",
                document_id=document.document_id,
                page_number=2,
                content_type=ContentType.TEXT,
                content="Remote work requires manager approval.",
                content_hash="element-hash-002",
            ),
        ]


class FakeChunker:
    def __init__(self) -> None:
        self.received_elements = None

    def chunk(
        self,
        elements,
    ):
        self.received_elements = elements

        return [
            DocumentChunk(
                chunk_id="chunk-001",
                document_id=elements[0].document_id,
                document_version_id="version-001",
                element_ids=[
                    elements[0].element_id,
                ],
                content=elements[0].content,
                chunk_index=0,
                content_hash="chunk-hash-001",
            ),
            DocumentChunk(
                chunk_id="chunk-002",
                document_id=elements[1].document_id,
                document_version_id="version-001",
                element_ids=[
                    elements[1].element_id,
                ],
                content=elements[1].content,
                chunk_index=1,
                content_hash="chunk-hash-002",
            ),
        ]


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.requests = []

    async def agenerate_batch(
        self,
        request,
    ):
        self.requests.append(request)

        return EmbeddingBatchResponse(
            vectors=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            model="fake-model",
            provider="fake-provider",
        )


class FakeSparseEmbeddingProvider:
    def __init__(self) -> None:
        self.text_batches = []

    async def agenerate_batch(
        self,
        texts,
    ):
        self.text_batches.append(texts)

        return [
            SparseEmbedding(
                indices=[1],
                values=[1.0],
            ),
            SparseEmbedding(
                indices=[2],
                values=[1.0],
            ),
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted_chunks = []

    async def aupsert(
        self,
        chunks: list[EmbeddedDocumentChunk],
    ) -> None:
        self.upserted_chunks.append(
            chunks
        )


def build_document() -> Document:
    now = datetime.now(
        timezone.utc
    )

    return Document(
        document_id="document-001",
        source="test.pdf",
        version=1,
        content_hash="document-hash-001",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_aingest_runs_complete_async_pipeline() -> None:
    storage = FakeStorageProvider()
    extractor = FakeExtractor()
    chunker = FakeChunker()
    embedder = FakeEmbeddingProvider()
    sparse_embedder = (
        FakeSparseEmbeddingProvider()
    )
    vector_store = FakeVectorStore()

    service = DocumentIngestionService(
        storage_provider=storage,
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        vector_store=vector_store,
    )

    document = build_document()
    version_id = uuid.uuid4()

    result = await service.aingest(
        document=document,
        blob_path="documents/document-001/v1/test.pdf",
        file_name="test.pdf",
        document_version_id=version_id,
        categories=["HR"],
        tags=["remote-work"],
    )

    assert storage.downloaded_paths == [
        "documents/document-001/v1/test.pdf"
    ]

    assert (
        extractor.received_file_name
        == "test.pdf"
    )

    assert (
        extractor.received_version_id
        == version_id
    )

    assert chunker.received_elements is not None

    assert len(
        chunker.received_elements
    ) == 2

    assert len(
        embedder.requests
    ) == 1

    assert embedder.requests[0].texts == [
        "Employees can work remotely.",
        "Remote work requires manager approval.",
    ]

    assert len(
        sparse_embedder.text_batches
    ) == 1

    assert sparse_embedder.text_batches[0] == [
        "Employees can work remotely.",
        "Remote work requires manager approval.",
    ]

    assert len(result) == 2

    assert result[0].vector == [
        1.0,
        0.0,
        0.0,
    ]

    assert result[1].vector == [
        0.0,
        1.0,
        0.0,
    ]

    assert result[0].sparse_vector.indices == [
        1
    ]

    assert result[1].sparse_vector.indices == [
        2
    ]

    for embedded_chunk in result:

        assert (
            embedded_chunk.chunk.metadata[
                "tenant_id"
            ]
        )

        assert (
            embedded_chunk.chunk.metadata[
                "categories"
            ]
            == ["HR"]
        )

        assert (
            embedded_chunk.chunk.metadata[
                "tags"
            ]
            == ["remote-work"]
        )

        assert (
            embedded_chunk.chunk.metadata[
                "document_version_id"
            ]
            == str(version_id)
        )

    assert len(
        vector_store.upserted_chunks
    ) == 1

    assert (
        vector_store.upserted_chunks[0]
        == result
    )


@pytest.mark.asyncio
async def test_aingest_returns_empty_when_no_chunks() -> None:
    storage = FakeStorageProvider()

    class EmptyExtractor(FakeExtractor):
        async def aextract(
            self,
            *,
            document,
            stream,
            file_name,
            document_version_id,
        ):
            return []

    class EmptyChunker(FakeChunker):
        def chunk(
            self,
            elements,
        ):
            self.received_elements = elements
            return []

    embedder = FakeEmbeddingProvider()

    sparse_embedder = (
        FakeSparseEmbeddingProvider()
    )

    vector_store = FakeVectorStore()

    service = DocumentIngestionService(
        storage_provider=storage,
        extractor=EmptyExtractor(),
        chunker=EmptyChunker(),
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        vector_store=vector_store,
    )

    result = await service.aingest(
        document=build_document(),
        blob_path="documents/test.pdf",
        file_name="test.pdf",
        document_version_id=uuid.uuid4(),
    )

    assert result == []

    assert embedder.requests == []

    assert sparse_embedder.text_batches == []

    assert vector_store.upserted_chunks == []