import pytest
from sqlalchemy import text

from app.db.session import AsyncSessionLocal


@pytest.mark.asyncio
async def test_async_database_connection() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT 1"))
        value = result.scalar_one()

    assert value == 1