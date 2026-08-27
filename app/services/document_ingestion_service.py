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
from app.providers.storage.base import StorageProvider
from app.providers.vector_store.base import VectorStore
from app.rag.chunking.base import DocumentChunker
from app.rag.ingestion.base import DocumentExtractor


class DocumentIngestionService:
    """
    Orchestrates the complete RAG ingestion pipeline.

    Flow:

        Blob Storage
            ->
        DocumentExtractor
            ->
        DocumentElement[]
            ->
        DocumentChunker
            ->
        DocumentChunk[]
            ->
        Metadata Enrichment
            ->
        EmbeddingProvider
            ->
        EmbeddedDocumentChunk[]
            ->
        VectorStore
            ->
        Qdrant

    The service orchestrates the pipeline but does not contain
    provider-specific Qdrant or embedding implementation details.
    """

    def __init__(
        self,
        storage_provider: StorageProvider,
        extractor: DocumentExtractor,
        chunker: DocumentChunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.storage_provider = storage_provider
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

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
        Run the complete ingestion pipeline.

        Steps:

            1. Download document
            2. Extract document elements
            3. Create document chunks
            4. Enrich chunk metadata
            5. Generate embeddings
            6. Pair chunks with vectors
            7. Store vectors and payload in VectorStore
        """

        # --------------------------------------------------
        # NORMALIZE DOCUMENT-LEVEL METADATA
        # --------------------------------------------------

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

        # Categories and tags are document-level metadata.
        # They do not influence chunk boundaries, so the chunker
        # should not know about them.
        #
        # We propagate them into chunk.metadata because the
        # vector store will use this metadata as Qdrant payload.

        for chunk in chunks:
            chunk.metadata["tenant_id"] = (
                chunk.metadata.get(
                    "tenant_id",
                    settings.default_tenant_id,
                )
            )

            chunk.metadata["categories"] = categories
            chunk.metadata["tags"] = tags

            # Keep version metadata explicit as well.
            chunk.metadata["document_version_id"] = str(
                document_version_id
            )

        # --------------------------------------------------
        # STEP 5 — EMBEDDING
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

        vectors = embedding_response.vectors

        # --------------------------------------------------
        # STEP 6 — PRESERVE CHUNK <-> VECTOR RELATIONSHIP
        # --------------------------------------------------

        embedded_chunks = [
            EmbeddedDocumentChunk(
                chunk=chunk,
                vector=vector,
            )
            for chunk, vector in zip(
                chunks,
                vectors,
                strict=True,
            )
        ]

        # --------------------------------------------------
        # STEP 7 — STORE IN VECTOR STORE
        # --------------------------------------------------

        self.vector_store.upsert(
            embedded_chunks
        )

        # Return the indexed objects for the caller/test layer.
        return embedded_chunks


# ============================================================
# ROHIT NOTES — INTERVIEW / CODE FLOW
# ============================================================

# 1. What is the responsibility of this service?
#
# It is the ORCHESTRATOR.
#
# It does not implement:
#
# - PDF extraction
# - chunking rules
# - embedding API logic
# - vector database logic
#
# Instead, it coordinates those components.
#
# Interview:
#
# "DocumentIngestionService orchestrates the ingestion pipeline,
# while extraction, chunking and embedding remain independently
# replaceable components."


# 2. Complete flow

# POST /documents
#        ↓
# DocumentService
#        ↓
# _upload_document_version()
#        ↓
# Blob Storage
#        ↓
# DocumentIngestionService
#        ↓
# Extraction
#        ↓
# DocumentElement[]
#        ↓
# HybridDocumentChunker
#        ↓
# DocumentChunk[]
#        ↓
# tenant/category/tag metadata
#        ↓
# Batch Embedding
#        ↓
# EmbeddedDocumentChunk[]
#        ↓
# VectorStore.upsert()
#        ↓
# Qdrant

# This is the current ContextOps ingestion pipeline.


# 3. Why download from Blob Storage again?
#
# The original upload stream may already have been consumed while
# calculating the hash and uploading the document.
#
# Blob Storage is therefore the source of truth for ingestion.


# 4. Why does the service not know Microsoft Foundry limits?
#
# It only depends on:
#
#     EmbeddingProvider
#
# MicrosoftFoundryEmbeddingProvider contains provider-specific
# limits such as:
#
# - maximum inputs
# - aggregate token limit
# - maximum tokens per input
#
# This keeps DocumentIngestionService provider-agnostic.


# 5. Why do we create EmbeddedDocumentChunk?
#
# Qdrant needs both:
#
#     vector
#     +
#     chunk metadata/content
#
# Example:
#
# Chunk:
# "Employees can work remotely up to 3 days."
#
# Vector:
# [0.12, -0.04, 0.81, ...]
#
# EmbeddedDocumentChunk keeps these together.


# 6. Why use zip(..., strict=True)?
#
# We expect exactly one embedding for every chunk.
#
# Example:
#
# chunks  = [C1, C2, C3]
# vectors = [V1, V2, V3]
#
# Result:
#
# C1 -> V1
# C2 -> V2
# C3 -> V3
#
# If the lengths differ, strict=True raises an error rather than
# silently producing incorrect mappings.


# 7. Why generate embeddings here?
#
# Chunking must happen before embedding because the embedding model
# should represent the final retrieval unit, not the entire document.
#
# Pipeline:
#
# Document
#   ↓
# Extract
#   ↓
# Chunk
#   ↓
# Embed
#   ↓
# Vector DB
#
# This matches the production RAG pattern of precomputing document
# embeddings during ingestion rather than generating them at query time.


# 8. What happens at query time later?
#
# Document ingestion:
#
# Chunk -> Embedding -> Qdrant
#
# User query:
#
# Query -> Query Embedding -> Qdrant similarity search
#
# The same embedding model must be used for both.


# 9. Current vs future responsibility
#
# CURRENT:
#
# Extraction
# Chunking
# Embedding
#
# NEXT:
#
# Qdrant indexing
#
# AFTER THAT:
#
# Retrieval
# Hybrid BM25 + Dense Search
# Reranking
# ContextOps
# LLM generation


# 10. Why is this architecture useful at large scale?
#
# Each stage can later be moved into independent workers:
#
# Upload API
#      ↓
# Ingestion Job
#      ↓
# Worker
#      ├── Extraction
#      ├── Chunking
#      ├── Batch Embedding
#      └── Qdrant indexing
#
# This avoids blocking API requests on expensive document processing.