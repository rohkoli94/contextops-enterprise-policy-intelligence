import uuid

from app.config.settings import settings
from app.domain.document import Document
from app.domain.document_chunk import DocumentChunk
from app.domain.document_element import DocumentElement
from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)
from app.providers.embedding.base import (
    EmbeddingBatchRequest,
    EmbeddingProvider,
)
from app.providers.embedding.sparse_base import (
    SparseEmbeddingProvider,
)
from app.providers.storage.base import StorageProvider
from app.providers.vector_store.base import VectorStore
from app.rag.chunking.base import DocumentChunker
from app.rag.ingestion.base import DocumentExtractor


class DocumentIngestionService:
    """
    Orchestrates the complete RAG ingestion pipeline.

    Supports both synchronous and asynchronous execution.

    Synchronous path:

        Blob Storage
            ->
        Extraction
            ->
        Chunking
            ->
        Metadata
            ->
        Dense Embeddings
            ->
        Sparse Embeddings
            ->
        Vector Store

    Asynchronous path:

        Async Blob Storage
            ->
        Async Extraction
            ->
        Chunking
            ->
        Metadata
            ->
        Async Dense Batch Embeddings
            ->
        Async Sparse Batch Embeddings
            ->
        Async Vector Store Upsert
    """

    def __init__(
        self,
        storage_provider: StorageProvider,
        extractor: DocumentExtractor,
        chunker: DocumentChunker,
        embedder: EmbeddingProvider,
        sparse_embedder: SparseEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.storage_provider = storage_provider
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.vector_store = vector_store

    # ============================================================
    # SYNCHRONOUS INGESTION
    # ============================================================

    def ingest(
        self,
        document: Document,
        blob_path: str,
        file_name: str,
        document_version_id: uuid.UUID,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[EmbeddedDocumentChunk]:
        """
        Run the complete ingestion pipeline synchronously.
        """

        categories = categories or []
        tags = tags or []

        # --------------------------------------------------
        # STEP 1 — DOWNLOAD
        # --------------------------------------------------

        stream = self.storage_provider.download(
            path=blob_path,
        )

        try:
            # --------------------------------------------------
            # STEP 2 — EXTRACTION
            # --------------------------------------------------

            elements: list[DocumentElement] = (
                self.extractor.extract(
                    document=document,
                    stream=stream,
                    file_name=file_name,
                    document_version_id=document_version_id,
                )
            )

        finally:
            stream.close()

        # --------------------------------------------------
        # STEP 3 — CHUNKING
        # --------------------------------------------------

        chunks: list[DocumentChunk] = (
            self.chunker.chunk(
                elements
            )
        )

        if not chunks:
            return []

        # --------------------------------------------------
        # STEP 4 — METADATA ENRICHMENT
        # --------------------------------------------------

        self._enrich_metadata(
            chunks=chunks,
            document_version_id=document_version_id,
            categories=categories,
            tags=tags,
        )

        # --------------------------------------------------
        # STEP 5 — DENSE EMBEDDING
        # --------------------------------------------------

        texts = [
            chunk.content
            for chunk in chunks
        ]

        embedding_response = (
            self.embedder.generate_batch(
                EmbeddingBatchRequest(
                    texts=texts,
                )
            )
        )

        dense_vectors = embedding_response.vectors

        # --------------------------------------------------
        # STEP 6 — SPARSE BM25 EMBEDDING
        # --------------------------------------------------

        sparse_vectors = (
            self.sparse_embedder.generate_batch(
                texts
            )
        )

        # --------------------------------------------------
        # STEP 7 — BUILD EMBEDDED CHUNKS
        # --------------------------------------------------

        embedded_chunks = (
            self._build_embedded_chunks(
                chunks=chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
            )
        )

        # --------------------------------------------------
        # STEP 8 — VECTOR STORE
        # --------------------------------------------------

        self.vector_store.upsert(
            embedded_chunks
        )

        return embedded_chunks

    # ============================================================
    # ASYNCHRONOUS INGESTION
    # ============================================================

    async def aingest(
        self,
        document: Document,
        blob_path: str,
        file_name: str,
        document_version_id: uuid.UUID,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[EmbeddedDocumentChunk]:
        """
        Run the complete ingestion pipeline asynchronously.

        The async path uses native async provider methods wherever
        available and keeps CPU-bound blocking work outside the
        event loop through provider implementations / async
        wrappers.
        """

        categories = categories or []
        tags = tags or []

        # --------------------------------------------------
        # STEP 1 — ASYNC DOWNLOAD
        # --------------------------------------------------

        stream = await self.storage_provider.adownload(
            path=blob_path,
        )

        try:
            # --------------------------------------------------
            # STEP 2 — ASYNC EXTRACTION
            # --------------------------------------------------

            elements: list[DocumentElement] = (
                await self.extractor.aextract(
                    document=document,
                    stream=stream,
                    file_name=file_name,
                    document_version_id=document_version_id,
                )
            )

        finally:
            stream.close()

        # --------------------------------------------------
        # STEP 3 — CHUNKING
        # --------------------------------------------------

        chunks: list[DocumentChunk] = (
            self.chunker.chunk(
                elements
            )
        )

        if not chunks:
            return []

        # --------------------------------------------------
        # STEP 4 — METADATA ENRICHMENT
        # --------------------------------------------------

        self._enrich_metadata(
            chunks=chunks,
            document_version_id=document_version_id,
            categories=categories,
            tags=tags,
        )

        # --------------------------------------------------
        # STEP 5 — ASYNC DENSE BATCH EMBEDDING
        # --------------------------------------------------

        texts = [
            chunk.content
            for chunk in chunks
        ]

        embedding_response = (
            await self.embedder.agenerate_batch(
                EmbeddingBatchRequest(
                    texts=texts,
                )
            )
        )

        dense_vectors = embedding_response.vectors

        # --------------------------------------------------
        # STEP 6 — ASYNC SPARSE BATCH EMBEDDING
        # --------------------------------------------------

        sparse_vectors = (
            await self.sparse_embedder.agenerate_batch(
                texts
            )
        )

        # --------------------------------------------------
        # STEP 7 — BUILD EMBEDDED CHUNKS
        # --------------------------------------------------

        embedded_chunks = (
            self._build_embedded_chunks(
                chunks=chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
            )
        )

        # --------------------------------------------------
        # STEP 8 — ASYNC VECTOR STORE UPSERT
        # --------------------------------------------------

        await self.vector_store.aupsert(
            embedded_chunks
        )

        return embedded_chunks

    # ============================================================
    # METADATA ENRICHMENT
    # ============================================================

    @staticmethod
    def _enrich_metadata(
        chunks: list[DocumentChunk],
        document_version_id: uuid.UUID,
        categories: list[str],
        tags: list[str],
    ) -> None:
        """
        Add document-level metadata to every chunk.
        """

        for chunk in chunks:

            chunk.metadata["tenant_id"] = (
                chunk.metadata.get(
                    "tenant_id",
                    settings.default_tenant_id,
                )
            )

            chunk.metadata["categories"] = (
                categories
            )

            chunk.metadata["tags"] = (
                tags
            )

            chunk.metadata["document_version_id"] = (
                str(document_version_id)
            )

    # ============================================================
    # BUILD EMBEDDED CHUNKS
    # ============================================================

    @staticmethod
    def _build_embedded_chunks(
        chunks: list[DocumentChunk],
        dense_vectors: list[list[float]],
        sparse_vectors,
    ) -> list[EmbeddedDocumentChunk]:
        """
        Pair each chunk with exactly one dense and one sparse
        representation.
        """

        return [
            EmbeddedDocumentChunk(
                chunk=chunk,
                vector=dense_vector,
                sparse_vector=sparse_vector,
            )
            for chunk, dense_vector, sparse_vector in zip(
                chunks,
                dense_vectors,
                sparse_vectors,
                strict=True,
            )
        ]