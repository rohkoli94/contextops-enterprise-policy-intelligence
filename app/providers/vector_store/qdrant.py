from typing import Any

from qdrant_client import (
    AsyncQdrantClient,
    QdrantClient,
    models,
)

from app.config.settings import settings
from app.domain.document_chunk import DocumentChunk
from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)
from app.domain.sparse_embedding import SparseEmbedding
from app.providers.vector_store.base import VectorStore
from app.rag.retrieval.models import RetrievedChunk


class QdrantVectorStore(VectorStore):
    """
    Qdrant implementation of the VectorStore abstraction.

    Architecture:

        One collection
            ->
        custom tenant sharding
            ->
        shared default shard
            +
        dedicated tenant shards
            ->
        payload filtering
            ->
        vector retrieval

    Payload contains retrieval and source metadata.

    The collection supports two named vector types:

        dense
            Semantic vector used for dense retrieval.

        bm25
            Sparse lexical vector used for BM25 retrieval.

    Retrieval is asynchronous.

    Administrative and write operations currently use the
    synchronous Qdrant client.
    """

    def __init__(self) -> None:
        # ----------------------------------------------------
        # SYNCHRONOUS CLIENT
        # ----------------------------------------------------
        #
        # Used for:
        #
        # - collection creation
        # - payload indexes
        # - shard management
        # - upsert
        #

        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

            self.async_client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
            )

            self.async_client = AsyncQdrantClient(
                url=settings.qdrant_url,
            )

    # ========================================================
    # COLLECTION
    # ========================================================

    def ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        """
        Create the Qdrant collection with custom sharding
        and support for both dense and sparse BM25 vectors.

        Dense vector:
            Used for semantic similarity retrieval.

        Sparse BM25 vector:
            Used for lexical / exact-term retrieval.

        In custom sharding mode, shard_number is the number
        of physical shards created per shard key.
        """

        collection_name = (
            settings.qdrant_collection_name
        )

        if self.client.collection_exists(
            collection_name=collection_name,
        ):
            return

        self.client.create_collection(
            collection_name=collection_name,

            # ------------------------------------------------
            # DENSE VECTOR
            # ------------------------------------------------

            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            },

            # ------------------------------------------------
            # SPARSE BM25 VECTOR
            # ------------------------------------------------

            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },

            # ------------------------------------------------
            # CUSTOM TENANT SHARDING
            # ------------------------------------------------

            shard_number=(
                settings.qdrant_shard_number
            ),
            sharding_method=(
                models.ShardingMethod.CUSTOM
            ),
        )

        self.ensure_payload_indexes()

        self.ensure_shard_key(
            settings.qdrant_default_shard_key
        )

    # ========================================================
    # PAYLOAD INDEXES
    # ========================================================

    def ensure_payload_indexes(self) -> None:
        """
        Create payload indexes for metadata used during
        retrieval filtering.

        Indexed fields:

        - tenant_id
        - document_id
        - document_version_id
        - categories
        - tags
        - content_type
        """

        collection_name = (
            settings.qdrant_collection_name
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="tenant_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_version_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="categories",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="tags",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="content_type",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

    # ========================================================
    # SHARD MANAGEMENT
    # ========================================================

    def ensure_shard_key(
        self,
        shard_key: str,
    ) -> None:
        """
        Ensure a custom shard key exists.

        Example:

            default
            large-bank
            large-insurer
        """

        existing_keys = self.client.list_shard_keys(
            collection_name=(
                settings.qdrant_collection_name
            ),
        )

        existing_key_values = {
            self._normalize_shard_key(key)
            for key in existing_keys
        }

        if shard_key in existing_key_values:
            return

        self.client.create_shard_key(
            collection_name=(
                settings.qdrant_collection_name
            ),
            shard_key=shard_key,
            shards_number=(
                settings.qdrant_shard_number
            ),
        )

    def promote_tenant_to_dedicated_shard(
        self,
        tenant_id: str,
    ) -> None:
        """
        Create a dedicated shard key for a tenant.

        Example:

            tenant_id = "large-bank"

            shard key = "large-bank"

        The actual decision to promote a tenant should be made
        by application/operations policy, based on workload,
        data size, query volume, isolation requirements, etc.
        """

        self.ensure_shard_key(
            tenant_id
        )

    # ========================================================
    # UPSERT
    # ========================================================

    def upsert(
        self,
        chunks: list[EmbeddedDocumentChunk],
    ) -> None:
        """
        Upsert embedded chunks into Qdrant.

        One request must contain chunks from exactly one tenant,
        because shard routing is tenant-specific.
        """

        if not chunks:
            return

        tenant_ids = {
            chunk.chunk.metadata.get(
                "tenant_id",
                settings.default_tenant_id,
            )
            for chunk in chunks
        }

        if len(tenant_ids) != 1:
            raise ValueError(
                "All chunks in one Qdrant upsert request "
                "must belong to the same tenant."
            )

        tenant_id = tenant_ids.pop()

        points = [
            self._to_point(
                embedded_chunk
            )
            for embedded_chunk in chunks
        ]

        # Tiered routing:
        #
        # target:
        #     tenant's dedicated shard, if it exists
        #
        # fallback:
        #     shared default shard
        #
        # Qdrant routes to target when active; otherwise
        # it uses fallback.

        shard_selector = (
            models.ShardKeyWithFallback(
                target=tenant_id,
                fallback=(
                    settings.qdrant_default_shard_key
                ),
            )
        )

        self.client.upsert(
            collection_name=(
                settings.qdrant_collection_name
            ),
            points=points,
            shard_key_selector=shard_selector,
            wait=True,
        )

    # ========================================================
    # DENSE SEARCH
    # ========================================================

    async def asearch_dense(
        self,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform asynchronous tenant-aware dense retrieval.

        Flow:

            Dense Query Vector
                ↓
            Tenant Shard Routing
                ↓
            Metadata Filter
                ↓
            Qdrant Dense ANN Search
                ↓
            RetrievedChunk[]
        """

        self._validate_dense_query(
            query_vector=query_vector,
            tenant_id=tenant_id,
            top_k=top_k,
        )

        query_filter = self._build_filter(
            tenant_id=tenant_id,
            filters=filters,
        )

        shard_selector = self.get_shard_selector(
            tenant_id=tenant_id,
        )

        response = await self.async_client.query_points(
            collection_name=(
                settings.qdrant_collection_name
            ),
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            shard_key_selector=shard_selector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return self._convert_results(
            response.points
        )

    # ========================================================
    # SPARSE / BM25 SEARCH
    # ========================================================

    async def asearch_sparse(
        self,
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform asynchronous tenant-aware sparse BM25 retrieval.

        Flow:

            Sparse BM25 Query
                ↓
            Tenant Shard Routing
                ↓
            Metadata Filter
                ↓
            Qdrant BM25 Search
                ↓
            RetrievedChunk[]
        """

        self._validate_sparse_query(
            sparse_query=sparse_query,
            tenant_id=tenant_id,
            top_k=top_k,
        )

        query_filter = self._build_filter(
            tenant_id=tenant_id,
            filters=filters,
        )

        shard_selector = self.get_shard_selector(
            tenant_id=tenant_id,
        )

        sparse_vector = models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        )

        # ----------------------------------------------------
        # TENANT-SCOPED IDF
        # ----------------------------------------------------

        tenant_idf_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id,
                    ),
                )
            ]
        )

        sparse_search_params = (
            models.SearchParams(
                idf=models.IdfCorpusParams(
                    corpus=tenant_idf_filter,
                ),
            )
        )

        response = await self.async_client.query_points(
            collection_name=(
                settings.qdrant_collection_name
            ),
            query=sparse_vector,
            using="bm25",
            query_filter=query_filter,
            shard_key_selector=shard_selector,
            params=sparse_search_params,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return self._convert_results(
            response.points
        )

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    async def asearch_hybrid(
        self,
        query_vector: list[float],
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform asynchronous tenant-aware hybrid retrieval.

        Dense and sparse candidate results are fused using
        Qdrant's native Reciprocal Rank Fusion.

        Flow:

            Dense Query
                  +
            BM25 Query
                  ↓
            Tenant Shard Routing
                  ↓
            Metadata Filtering
                  ↓
            ┌──────────────────────┐
            │ Dense candidates     │
            │ BM25 candidates      │
            └──────────┬───────────┘
                       ↓
                      RRF
                       ↓
                RetrievedChunk[]
        """

        self._validate_dense_query(
            query_vector=query_vector,
            tenant_id=tenant_id,
            top_k=top_k,
        )

        self._validate_sparse_query(
            sparse_query=sparse_query,
            tenant_id=tenant_id,
            top_k=top_k,
        )

        query_filter = self._build_filter(
            tenant_id=tenant_id,
            filters=filters,
        )

        shard_selector = self.get_shard_selector(
            tenant_id=tenant_id,
        )

        sparse_vector = models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        )

        # ----------------------------------------------------
        # TENANT-SCOPED IDF
        # ----------------------------------------------------

        tenant_idf_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id,
                    ),
                )
            ]
        )

        sparse_search_params = (
            models.SearchParams(
                idf=models.IdfCorpusParams(
                    corpus=tenant_idf_filter,
                ),
            )
        )

        # ----------------------------------------------------
        # QDRANT HYBRID SEARCH
        # ----------------------------------------------------

        response = await self.async_client.query_points(
            collection_name=(
                settings.qdrant_collection_name
            ),

            # ------------------------------------------------
            # CANDIDATE RETRIEVAL
            # ------------------------------------------------

            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    filter=query_filter,
                    limit=top_k,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="bm25",
                    filter=query_filter,
                    params=sparse_search_params,
                    limit=top_k,
                ),
            ],

            # ------------------------------------------------
            # RECIPROCAL RANK FUSION
            # ------------------------------------------------

            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),

            # ------------------------------------------------
            # TENANT SHARD ROUTING
            # ------------------------------------------------

            shard_key_selector=shard_selector,

            # ------------------------------------------------
            # FINAL RESULT SIZE
            # ------------------------------------------------

            limit=top_k,

            with_payload=True,
            with_vectors=False,
        )

        return self._convert_results(
            response.points
        )

    # ========================================================
    # FILTER BUILDER
    # ========================================================

    def _build_filter(
        self,
        tenant_id: str,
        filters: dict[str, Any] | None,
    ) -> models.Filter:
        """
        Build the Qdrant metadata filter.

        tenant_id is always mandatory.

        Supported optional filters:

            document_id
            document_version_id
            categories
            tags
            content_type
        """

        conditions: list[
            models.FieldCondition
        ] = []

        # --------------------------------------------------
        # TENANT FILTER
        # --------------------------------------------------

        conditions.append(
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=tenant_id,
                ),
            )
        )

        # --------------------------------------------------
        # OPTIONAL FILTERS
        # --------------------------------------------------

        if filters:

            allowed_fields = {
                "document_id",
                "document_version_id",
                "categories",
                "tags",
                "content_type",
            }

            for key, value in filters.items():

                if key not in allowed_fields:
                    raise ValueError(
                        f"Unsupported retrieval filter: "
                        f"{key}"
                    )

                if value is None:
                    continue

                # ------------------------------------------
                # SINGLE VALUE
                # ------------------------------------------

                if isinstance(value, str):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(
                                value=value,
                            ),
                        )
                    )

                # ------------------------------------------
                # MULTIPLE VALUES
                # ------------------------------------------

                elif isinstance(value, list):

                    if not value:
                        continue

                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(
                                any=value,
                            ),
                        )
                    )

                else:
                    raise ValueError(
                        f"Unsupported filter value "
                        f"for field '{key}'"
                    )

        return models.Filter(
            must=conditions,
        )

    # ========================================================
    # RESULT CONVERSION
    # ========================================================

    def _convert_results(
        self,
        points: list[Any],
    ) -> list[RetrievedChunk]:
        """
        Convert Qdrant scored points into RetrievedChunk
        domain results.
        """

        results: list[RetrievedChunk] = []

        for point in points:

            payload = point.payload or {}

            chunk = (
                self._payload_to_document_chunk(
                    payload=payload,
                )
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(
                        point.score
                    ),
                    metadata=payload,
                )
            )

        return results

    # ========================================================
    # PAYLOAD -> DOCUMENT CHUNK
    # ========================================================

    def _payload_to_document_chunk(
        self,
        payload: dict[str, Any],
    ) -> DocumentChunk:
        """
        Reconstruct a DocumentChunk from Qdrant payload.

        The Qdrant payload contains the information required
        to return a retrieval-ready chunk without fetching
        PostgreSQL data for every search result.
        """

        return DocumentChunk(
            chunk_id=str(
                payload["chunk_id"]
            ),
            document_id=str(
                payload["document_id"]
            ),
            document_version_id=str(
                payload["document_version_id"]
            ),
            element_ids=[
                str(element_id)
                for element_id in payload.get(
                    "element_ids",
                    [],
                )
            ],
            content=str(
                payload["content"]
            ),
            chunk_index=int(
                payload.get(
                    "chunk_index",
                    0,
                )
            ),
            content_hash=str(
                payload.get(
                    "content_hash",
                    "",
                )
            ),
            metadata={
                "tenant_id": payload.get(
                    "tenant_id"
                ),
                "categories": payload.get(
                    "categories",
                    [],
                ),
                "tags": payload.get(
                    "tags",
                    [],
                ),
                "content_type": payload.get(
                    "content_type"
                ),
                "hierarchy_path": payload.get(
                    "hierarchy_path",
                    [],
                ),
                "page_numbers": payload.get(
                    "page_numbers",
                    [],
                ),
            },
        )

    # ========================================================
    # POINT MAPPING
    # ========================================================

    def _to_point(
        self,
        embedded_chunk: EmbeddedDocumentChunk,
    ) -> models.PointStruct:
        """
        Convert EmbeddedDocumentChunk into a Qdrant point.

        The Qdrant point contains:

        - dense vector for semantic retrieval
        - sparse BM25 vector for lexical retrieval
        - payload containing document and retrieval metadata
        """

        chunk = embedded_chunk.chunk
        metadata = chunk.metadata

        payload = {
            # ------------------------------------------------
            # Tenant
            # ------------------------------------------------

            "tenant_id": metadata.get(
                "tenant_id",
                settings.default_tenant_id,
            ),

            # ------------------------------------------------
            # Identity
            # ------------------------------------------------

            "chunk_id": str(
                chunk.chunk_id
            ),

            "document_id": str(
                chunk.document_id
            ),

            "document_version_id": str(
                chunk.document_version_id
            ),

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            "content": chunk.content,

            # ------------------------------------------------
            # Lineage
            # ------------------------------------------------

            "element_ids": [
                str(element_id)
                for element_id in chunk.element_ids
            ],

            # ------------------------------------------------
            # Chunk position / deduplication
            # ------------------------------------------------

            "chunk_index": chunk.chunk_index,

            "content_hash": chunk.content_hash,

            # ------------------------------------------------
            # Content metadata
            # ------------------------------------------------

            "content_type": metadata.get(
                "content_type"
            ),

            "hierarchy_path": metadata.get(
                "hierarchy_path",
                [],
            ),

            "page_numbers": metadata.get(
                "page_numbers",
                [],
            ),

            # ------------------------------------------------
            # Retrieval metadata
            # ------------------------------------------------

            "categories": metadata.get(
                "categories",
                [],
            ),

            "tags": metadata.get(
                "tags",
                [],
            ),
        }

        # ----------------------------------------------------
        # SPARSE BM25 VECTOR
        # ----------------------------------------------------

        sparse_vector = models.SparseVector(
            indices=embedded_chunk.sparse_vector.indices,
            values=embedded_chunk.sparse_vector.values,
        )

        # ----------------------------------------------------
        # QDRANT POINT
        # ----------------------------------------------------

        return models.PointStruct(
            id=str(
                chunk.chunk_id
            ),
            vector={
                "dense": embedded_chunk.vector,
                "bm25": sparse_vector,
            },
            payload=payload,
        )

    # ========================================================
    # SHARD SELECTOR
    # ========================================================

    def get_shard_selector(
        self,
        tenant_id: str,
    ) -> models.ShardKeyWithFallback:
        """
        Build the shard selector used during retrieval.

        target:
            dedicated tenant shard

        fallback:
            shared default shard
        """

        return models.ShardKeyWithFallback(
            target=tenant_id,
            fallback=(
                settings.qdrant_default_shard_key
            ),
        )

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    @staticmethod
    def _validate_dense_query(
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
    ) -> None:
        """
        Validate dense retrieval inputs.
        """

        if not query_vector:
            raise ValueError(
                "Query vector cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

    @staticmethod
    def _validate_sparse_query(
        sparse_query: SparseEmbedding,
        tenant_id: str,
        top_k: int,
    ) -> None:
        """
        Validate sparse BM25 retrieval inputs.
        """

        if not sparse_query.indices:
            raise ValueError(
                "Sparse BM25 query cannot be empty."
            )

        if len(sparse_query.indices) != len(
            sparse_query.values
        ):
            raise ValueError(
                "Sparse query indices and values "
                "must have the same length."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "Tenant ID cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

    # ========================================================
    # ASYNC CLIENT LIFECYCLE
    # ========================================================

    async def aclose(self) -> None:
        """
        Close the asynchronous Qdrant client.

        Called during FastAPI application shutdown.
        """

        await self.async_client.close()


    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize_shard_key(
        shard_key: object,
    ) -> str:
        """
        Normalize Qdrant's returned shard-key representation
        into a comparable string.
        """

        if isinstance(shard_key, str):
            return shard_key

        return str(shard_key)


# ============================================================
# ROHIT NOTES — INTERVIEW / CODE FLOW
# ============================================================

# So we have two layers of protection/optimization:

# Shard routing
# → physically narrow the area we query

# Metadata filtering
# → logically narrow the matching documents/chunks


# ============================================================
# RETRIEVAL MODES
# ============================================================

# Dense:
#
# Query
#   ↓
# Dense embedding
#   ↓
# asearch_dense()
#   ↓
# Qdrant dense ANN
#   ↓
# RetrievedChunk[]


# Sparse:
#
# Query
#   ↓
# BM25 sparse representation
#   ↓
# asearch_sparse()
#   ↓
# Qdrant BM25
#   ↓
# RetrievedChunk[]


# Hybrid:
#
# Query
#   ↓
# ┌───────────────────┐
# │ Dense query        │
# │ BM25 sparse query  │
# └─────────┬─────────┘
#           ↓
# asearch_hybrid()
#           ↓
# Dense candidates
# +
# BM25 candidates
#           ↓
#          RRF
#           ↓
# RetrievedChunk[]