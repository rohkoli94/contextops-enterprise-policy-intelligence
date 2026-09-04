import pytest

from app.guardrails.authorization import AuthorizationGuard


@pytest.mark.asyncio
async def test_authorization_allows_valid_tenant() -> None:
    guard = AuthorizationGuard()

    result = await guard.evaluate(
        tenant_id="tenant-001",
        authorized_tenant_ids={"tenant-001"},
    )

    assert result.allowed is True
    assert result.action.value == "allow"


@pytest.mark.asyncio
async def test_authorization_blocks_unauthorized_tenant() -> None:
    guard = AuthorizationGuard()

    result = await guard.evaluate(
        tenant_id="tenant-002",
        authorized_tenant_ids={"tenant-001"},
    )

    assert result.allowed is False
    assert result.action.value == "block"


@pytest.mark.asyncio
async def test_authorization_blocks_missing_tenant() -> None:
    guard = AuthorizationGuard()

    result = await guard.evaluate(tenant_id="")

    assert result.allowed is False
    assert result.action.value == "block"