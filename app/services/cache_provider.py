from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    """
    Cached query response.

    The payload is intentionally generic because different cache
    implementations may store different response representations.
    """

    value: dict[str, Any]
    cache_key: str


class CacheProvider(ABC):
    """
    Application-level cache abstraction.

    Production implementation will be backed by Redis.

    The workflow must continue to work when cache is unavailable;
    caching is an optimization and must not become a correctness
    dependency.
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