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
        Dense EmbeddingProvider
            +
        Sparse EmbeddingProvider
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
        sparse_embedder: SparseEmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.storage_provider = storage_provider
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
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
            5. Generate dense embeddings
            6. Generate sparse BM25 embeddings
            7. Pair chunks with both vector representations
            8. Store vectors and payload in VectorStore
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

        # BM25 does not use the dense embedding model.
        #
        # It creates a sparse lexical representation based on
        # the terms present in the chunk text.

        sparse_vectors = (
            self.sparse_embedder.generate_batch(
                texts
            )
        )

        # --------------------------------------------------
        # STEP 7 — PRESERVE CHUNK/VECTOR RELATIONSHIPS
        # --------------------------------------------------

        # Every chunk must have:
        #
        #     one dense vector
        #     +
        #     one sparse BM25 vector
        #
        # Chunk 0 -> Dense 0 -> Sparse 0
        # Chunk 1 -> Dense 1 -> Sparse 1
        # Chunk 2 -> Dense 2 -> Sparse 2
        #
        # strict=True protects the one-to-one relationship.

        embedded_chunks = [
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

        # --------------------------------------------------
        # STEP 8 — STORE IN VECTOR STORE
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
#
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
# Dense Embedding
#        ↓
# Sparse BM25 Embedding
#        ↓
# EmbeddedDocumentChunk[]
#        ↓
# VectorStore.upsert()
#        ↓
# Qdrant
#
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
# Qdrant needs:
#
#     dense vector
#     +
#     sparse BM25 vector
#     +
#     chunk metadata/content
#
# Example:
#
# Chunk:
# "Employees can work remotely up to 3 days."
#
# Dense Vector:
# [0.12, -0.04, 0.81, ...]
#
# Sparse BM25 Vector:
# indices = [...]
# values  = [...]
#
# EmbeddedDocumentChunk keeps these retrieval
# representations together with the original chunk.


# 6. Why use zip(..., strict=True)?
#
# We expect exactly one dense embedding and one sparse
# embedding for every chunk.
#
# Example:
#
# chunks        = [C1, C2, C3]
# dense_vectors = [D1, D2, D3]
# sparse_vectors = [S1, S2, S3]
#
# Result:
#
# C1 -> D1 -> S1
# C2 -> D2 -> S2
# C3 -> D3 -> S3
#
# If any lengths differ, strict=True raises an error rather
# than silently producing incorrect mappings.


# 7. Why generate embeddings here?
#
# Chunking must happen before embedding because the embedding
# models should represent the final retrieval unit, not the
# entire document.
#
# Pipeline:
#
# Document
#   ↓
# Extract
#   ↓
# Chunk
#   ↓
# Metadata enrichment
#   ↓
# Dense embedding + Sparse BM25 embedding
#   ↓
# Vector DB
#
# Dense retrieval uses semantic embeddings.
#
# BM25 retrieval uses sparse lexical representations.
#
# Both are precomputed during ingestion.


# 8. What happens at query time later?
#
# Document ingestion:
#
# Chunk
#   ├── Dense embedding
#   └── BM25 sparse representation
#          ↓
#       Qdrant
#
# User query:
#
# Query
#   ├── Dense query embedding
#   └── BM25 sparse query representation
#          ↓
#       Qdrant
#          ↓
#    Dense + BM25 results
#          ↓
#         RRF


# 9. Current vs future responsibility
#
# CURRENT:
#
# Extraction
# Chunking
# Metadata enrichment
# Dense embedding
# Sparse BM25 embedding
# Qdrant indexing
#
# NEXT:
#
# Retrieval
# Hybrid BM25 + Dense Search
# RRF
# Reranking
# Query intelligence
# LangGraph orchestration
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
#      ├── Metadata enrichment
#      ├── Dense embedding
#      ├── Sparse BM25 embedding
#      └── Qdrant indexing
#
# This avoids blocking API requests on expensive document processing.