from collections.abc import Callable

from app.guardrails import (
    AuthorizationGuard,
    PIIAnalyzer,
    PromptInjectionGuard,
    RegexPIIAnalyzer,
    TenantIsolationGuard,
)
from app.rag.workflow.state import QueryState


class SecurityGuardrails:
    """
    Coordinates all request-level security guardrails.
    """

    def __init__(
        self,
        authorization_guard: AuthorizationGuard,
        tenant_isolation_guard: TenantIsolationGuard,
        prompt_injection_guard: PromptInjectionGuard,
        pii_analyzer: PIIAnalyzer,
    ) -> None:
        self.authorization_guard = authorization_guard
        self.tenant_isolation_guard = tenant_isolation_guard
        self.prompt_injection_guard = prompt_injection_guard
        self.pii_analyzer = pii_analyzer

    async def evaluate(self, state: QueryState) -> QueryState:
        query = state["query"]
        tenant_id = state["tenant_id"]

        authorization_result = await self.authorization_guard.evaluate(
            tenant_id=tenant_id,
        )

        if not authorization_result.allowed:
            raise PermissionError(
                authorization_result.reason
                or "Authorization failed."
            )

        tenant_result = await self.tenant_isolation_guard.evaluate(
            tenant_id=tenant_id,
        )

        if not tenant_result.allowed:
            raise PermissionError(
                tenant_result.reason
                or "Tenant isolation validation failed."
            )

        injection_result = await self.prompt_injection_guard.evaluate(
            text=query,
        )

        if not injection_result.allowed:
            raise ValueError(
                injection_result.reason
                or "Potential prompt injection detected."
            )

        pii_result = await self.pii_analyzer.analyze(query)

        return {
            **state,
            "authorization_valid": True,
            "tenant_valid": True,
            "prompt_injection_detected": False,
            "input_pii_detected": pii_result.detected,
            "input_guardrail_action": pii_result.action.value,
            "guardrail_reason": None,
        }


def create_security_guardrails_node(
    authorization_guard: AuthorizationGuard,
    tenant_isolation_guard: TenantIsolationGuard,
    prompt_injection_guard: PromptInjectionGuard,
    pii_analyzer: PIIAnalyzer,
) -> Callable[[QueryState], object]:
    """
    Creates a LangGraph node using shared guardrail dependencies.
    """

    security_guardrails = SecurityGuardrails(
        authorization_guard=authorization_guard,
        tenant_isolation_guard=tenant_isolation_guard,
        prompt_injection_guard=prompt_injection_guard,
        pii_analyzer=pii_analyzer,
    )

    async def node(state: QueryState) -> QueryState:
        return await security_guardrails.evaluate(state)

    return node