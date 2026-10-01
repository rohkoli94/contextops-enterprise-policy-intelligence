from typing import Any

from app.services.cache_provider import (
    CacheEntry,
    CacheProvider,
)


class NoOpCacheProvider(CacheProvider):
    """Cache implementation that always returns a miss."""

    async def get(self, key: str) -> CacheEntry | None:
        return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def get_knowledge_base_version(self, tenant_id: str) -> str:
        return "0"

    async def bump_knowledge_base_version(self, tenant_id: str) -> str:
        return "0"
