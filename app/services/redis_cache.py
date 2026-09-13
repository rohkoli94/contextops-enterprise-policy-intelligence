import json
from typing import Any

from redis.asyncio import Redis

from app.services.cache_provider import (
    CacheEntry,
    CacheProvider,
)


class RedisCacheProvider(CacheProvider):
    """
    Redis-backed asynchronous cache provider.

    The provider stores serialized Python dictionaries as JSON
    strings and applies a per-entry TTL.

    Redis is treated as an optimization layer:
    cache failures should be handled by the workflow as cache misses.
    """

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

    # ========================================================
    # KEY
    # ========================================================

    def _build_redis_key(
        self,
        key: str,
    ) -> str:
        return f"{self.key_prefix}{key}"

    # ========================================================
    # GET
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
            return None

        payload = json.loads(
            value
        )

        if not isinstance(payload, dict):
            raise ValueError(
                "Cached Redis value must deserialize to a dictionary."
            )

        return CacheEntry(
            value=payload,
            cache_key=key,
        )

    # ========================================================
    # SET
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

    # ========================================================
    # DELETE
    # ========================================================

    async def delete(
        self,
        key: str,
    ) -> None:
        redis_key = self._build_redis_key(
            key
        )

        await self.redis.delete(
            redis_key
        )

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self) -> None:
        """
        Close the Redis connection pool.
        """

        await self.redis.aclose()