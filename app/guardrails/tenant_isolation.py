from app.guardrails.base import Guardrail
from app.guardrails.models import (
    GuardrailAction,
    GuardrailResult,
    GuardrailType,
)


class TenantIsolationGuard(Guardrail):
    """
    Enforces the mandatory tenant boundary.

    This guard is intentionally strict.

    Important production rule:
    retrieval recovery, query rewriting, broader retrieval,
    or cache fallback must NEVER remove tenant isolation.
    """

    async def evaluate(
        self,
        *,
        tenant_id: str,
        **_: object,
    ) -> GuardrailResult:

        if not tenant_id or not tenant_id.strip():
            return GuardrailResult(
                allowed=False,
                action=GuardrailAction.BLOCK,
                guardrail_type=GuardrailType.TENANT_ISOLATION,
                reason="tenant_id is required.",
            )

        return GuardrailResult(
            allowed=True,
            action=GuardrailAction.ALLOW,
            guardrail_type=GuardrailType.TENANT_ISOLATION,
            metadata={
                "tenant_id": tenant_id,
            },
        )