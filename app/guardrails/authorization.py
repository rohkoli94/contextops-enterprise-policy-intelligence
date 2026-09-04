from app.guardrails.base import Guardrail
from app.guardrails.models import (
    GuardrailAction,
    GuardrailResult,
    GuardrailType,
)


class AuthorizationGuard(Guardrail):
    """
    Validates whether the caller is authorized to access the
    requested tenant.

    Day 18 provides the application-level contract.

    Production integration can later connect this to:
    - Microsoft Entra ID
    - JWT claims
    - RBAC
    - ABAC
    - API gateway authorization

    The LangGraph workflow should not need to change when the
    underlying authorization implementation changes.
    """

    async def evaluate(
        self,
        *,
        tenant_id: str,
        authorized_tenant_ids: set[str] | None = None,
        **_: object,
    ) -> GuardrailResult:

        if not tenant_id or not tenant_id.strip():
            return GuardrailResult(
                allowed=False,
                action=GuardrailAction.BLOCK,
                guardrail_type=GuardrailType.AUTHORIZATION,
                reason="Tenant identity is missing.",
            )

        # Prototype/application boundary:
        # when an explicit authorization set is provided,
        # enforce it strictly.
        if (
            authorized_tenant_ids is not None
            and tenant_id not in authorized_tenant_ids
        ):
            return GuardrailResult(
                allowed=False,
                action=GuardrailAction.BLOCK,
                guardrail_type=GuardrailType.AUTHORIZATION,
                reason="Caller is not authorized for the requested tenant.",
                metadata={
                    "tenant_id": tenant_id,
                },
            )

        return GuardrailResult(
            allowed=True,
            action=GuardrailAction.ALLOW,
            guardrail_type=GuardrailType.AUTHORIZATION,
            metadata={
                "tenant_id": tenant_id,
            },
        )