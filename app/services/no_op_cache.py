from typing import Any

from app.services.cache_provider import (
    CacheEntry,
    CacheProvider,
)


class NoOpCacheProvider(CacheProvider):
    """
    Day 18 cache implementation.

    Always returns a cache miss.

    This provides a valid application implementation while
    keeping Redis-specific infrastructure for the later
    production cache implementation.
    """

    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        return None

    async def delete(
        self,
        key: str,
    ) -> None:
        return None