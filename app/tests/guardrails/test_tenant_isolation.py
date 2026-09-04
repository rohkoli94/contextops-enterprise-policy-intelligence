import pytest

from app.guardrails.tenant_isolation import TenantIsolationGuard


@pytest.mark.asyncio
async def test_tenant_isolation_allows_valid_tenant() -> None:
    guard = TenantIsolationGuard()

    result = await guard.evaluate(
        tenant_id="tenant-001",
    )

    assert result.allowed is True
    assert result.action.value == "allow"


@pytest.mark.asyncio
async def test_tenant_isolation_blocks_missing_tenant() -> None:
    guard = TenantIsolationGuard()

    result = await guard.evaluate(
        tenant_id="",
    )

    assert result.allowed is False
    assert result.action.value == "block"