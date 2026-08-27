from qdrant_client import QdrantClient, models

from app.config.settings import settings
from app.domain.embedded_document_chunk import (
    EmbeddedDocumentChunk,
)
from app.providers.vector_store.base import VectorStore


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
    """

    def __init__(self) -> None:
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            self.client = QdrantClient(
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
        if it does not already exist.

        In custom sharding mode, shard_number is the number
        of physical shards created per shard key.
        """

        collection_name = (
            settings.qdrant_collection_name
        )

        if not self.client.collection_exists(
            collection_name=collection_name,
        ):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                shard_number=settings.qdrant_shard_number,
                sharding_method=models.ShardingMethod.CUSTOM,
            )

        # Keep initialization idempotent at application level.
        self.ensure_payload_indexes()

        # Ensure the shared fallback shard exists.
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
    # POINT MAPPING
    # ========================================================

    def _to_point(
        self,
        embedded_chunk: EmbeddedDocumentChunk,
    ) -> models.PointStruct:
        """
        Convert EmbeddedDocumentChunk into a Qdrant point.

        The existing DocumentChunk.metadata is the single
        metadata source for the Qdrant payload.
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

        return models.PointStruct(
            id=str(
                chunk.chunk_id
            ),
            vector=embedded_chunk.vector,
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