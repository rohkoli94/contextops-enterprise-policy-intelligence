from unittest.mock import AsyncMock

import pytest

from app.services.redis_cache import RedisCacheProvider


@pytest.mark.asyncio
async def test_missing_generation_defaults_to_zero() -> None:
    provider = RedisCacheProvider(
        redis_url="redis://localhost:6379/0",
    )
    provider.redis.get = AsyncMock(return_value=None)

    value = await provider.get_knowledge_base_version("contextops")

    assert value == "0"
    provider.redis.get.assert_awaited_once_with(
        "contextops:cache:kb-version:contextops"
    )

    await provider.close()


@pytest.mark.asyncio
async def test_generation_is_read_as_string() -> None:
    provider = RedisCacheProvider(
        redis_url="redis://localhost:6379/0",
    )
    provider.redis.get = AsyncMock(return_value="17")

    value = await provider.get_knowledge_base_version("contextops")

    assert value == "17"
    await provider.close()


@pytest.mark.asyncio
async def test_generation_bump_is_atomic_increment() -> None:
    provider = RedisCacheProvider(
        redis_url="redis://localhost:6379/0",
    )
    provider.redis.incr = AsyncMock(return_value=18)

    value = await provider.bump_knowledge_base_version("contextops")

    assert value == "18"
    provider.redis.incr.assert_awaited_once_with(
        "contextops:cache:kb-version:contextops"
    )
    await provider.close()
