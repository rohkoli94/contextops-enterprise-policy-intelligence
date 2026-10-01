from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    """A cached application response."""

    value: dict[str, Any]
    cache_key: str


class CacheProvider(ABC):
    """
    Application-level cache abstraction.

    Cache is an optimization and must never be a correctness
    dependency for the RAG workflow.

    The two knowledge-base generation methods are intentionally
    concrete defaults so existing test doubles and non-Redis cache
    implementations remain source-compatible.
    """

    @abstractmethod
    async def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        raise NotImplementedError

    @abstractmethod
    async def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        key: str,
    ) -> None:
        raise NotImplementedError

    async def get_knowledge_base_version(
        self,
        tenant_id: str,
    ) -> str:
        """Return the default generation for non-persistent caches."""
        return "0"

    async def bump_knowledge_base_version(
        self,
        tenant_id: str,
    ) -> str:
        """Return the default generation for non-persistent caches."""
        return "0"
