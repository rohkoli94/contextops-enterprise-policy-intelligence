import json
from typing import Any

from redis.asyncio import Redis

from app.core.logging import get_logger
from app.services.cache_provider import (
    CacheEntry,
    CacheProvider,
)


logger = get_logger(__name__)


class RedisCacheProvider(CacheProvider):
    """
    Redis-backed asynchronous cache provider.

    Redis stores both query responses and the tenant-scoped
    knowledge-base generation used for cache invalidation.

    The generation is monotonic and is changed with Redis INCR, so
    all new query cache keys immediately move to the new namespace.
    Existing old-generation entries are left to expire through the
    normal TTL instead of scanning or deleting them.
    """

    KNOWLEDGE_BASE_VERSION_KEY_PREFIX = "kb-version:"
    INITIAL_KNOWLEDGE_BASE_VERSION = "0"

    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str = "contextops:cache:",
    ) -> None:
        self.key_prefix = key_prefix

        self.redis: Redis = Redis.from_url(
            redis_url,
            decode_responses=True,
        )

        logger.debug(
            "Initialized Redis cache provider",
            extra={
                "key_prefix": self.key_prefix,
            },
        )

    # ========================================================
    # KEY
    # ========================================================

    def _build_redis_key(
        self,
        key: str,
    ) -> str:
        return f"{self.key_prefix}{key}"

    def _build_knowledge_base_version_key(
        self,
        tenant_id: str,
    ) -> str:
        normalized_tenant_id = tenant_id.strip()

        if not normalized_tenant_id:
            raise ValueError(
                "tenant_id must not be empty."
            )

        return self._build_redis_key(
            f"{self.KNOWLEDGE_BASE_VERSION_KEY_PREFIX}"
            f"{normalized_tenant_id}"
        )

    # ========================================================
    # GET QUERY CACHE
    # ========================================================

    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        redis_key = self._build_redis_key(
            key
        )

        value = await self.redis.get(
            redis_key
        )

        if value is None:
            logger.debug(
                "Redis query cache miss",
                extra={
                    "cache_key": key,
                },
            )
            return None

        payload = json.loads(
            value
        )

        if not isinstance(payload, dict):
            raise ValueError(
                "Cached Redis value must deserialize to a dictionary."
            )

        logger.debug(
            "Redis query cache hit",
            extra={
                "cache_key": key,
            },
        )

        return CacheEntry(
            value=payload,
            cache_key=key,
        )

    # ========================================================
    # SET QUERY CACHE
    # ========================================================

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than zero."
            )

        redis_key = self._build_redis_key(
            key
        )

        serialized_value = json.dumps(
            value,
            default=str,
        )

        await self.redis.set(
            redis_key,
            serialized_value,
            ex=ttl_seconds,
        )

        logger.debug(
            "Stored Redis query cache entry",
            extra={
                "cache_key": key,
                "ttl_seconds": ttl_seconds,
            },
        )

    # ========================================================
    # DELETE QUERY CACHE
    # ========================================================

    async def delete(
        self,
        key: str,
    ) -> None:
        redis_key = self._build_redis_key(
            key
        )

        deleted = await self.redis.delete(
            redis_key
        )

        logger.debug(
            "Deleted Redis query cache entry",
            extra={
                "cache_key": key,
                "deleted": bool(deleted),
            },
        )

    # ========================================================
    # KNOWLEDGE-BASE VERSION
    # ========================================================

    async def get_knowledge_base_version(
        self,
        tenant_id: str,
    ) -> str:
        """
        Return the tenant-scoped knowledge-base generation.

        A missing generation starts at version ``0`` without an
        explicit Redis write. The first successful corpus change
        increments it to ``1``.
        """

        redis_key = self._build_knowledge_base_version_key(
            tenant_id
        )

        value = await self.redis.get(
            redis_key
        )

        if value is None:
            version = self.INITIAL_KNOWLEDGE_BASE_VERSION

            logger.info(
                "Knowledge-base cache generation not initialized; using default",
                extra={
                    "tenant_id": tenant_id,
                    "knowledge_base_version": version,
                },
            )

            return version

        version = str(value)

        logger.debug(
            "Read knowledge-base cache generation",
            extra={
                "tenant_id": tenant_id,
                "knowledge_base_version": version,
            },
        )

        return version

    async def bump_knowledge_base_version(
        self,
        tenant_id: str,
    ) -> str:
        """
        Atomically increment the tenant-scoped knowledge-base
        generation and return the new value.
        """

        redis_key = self._build_knowledge_base_version_key(
            tenant_id
        )

        version = await self.redis.incr(
            redis_key
        )

        version_string = str(version)

        logger.info(
            "Bumped knowledge-base cache generation",
            extra={
                "tenant_id": tenant_id,
                "knowledge_base_version": version_string,
            },
        )

        return version_string

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self) -> None:
        """
        Close the Redis connection pool.
        """

        await self.redis.aclose()

        logger.debug(
            "Closed Redis cache provider"
        )
