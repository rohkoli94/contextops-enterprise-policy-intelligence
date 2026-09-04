from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    MASK = "mask"
    AUDIT = "audit"


class GuardrailType(StrEnum):
    AUTHORIZATION = "authorization"
    TENANT_ISOLATION = "tenant_isolation"
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    OUTPUT = "output"


@dataclass(frozen=True)
class GuardrailResult:
    """
    Result returned by a guardrail evaluation.

    `allowed` determines whether the workflow may continue.
    `action` describes what should happen with the input/output.
    """

    allowed: bool
    action: GuardrailAction
    guardrail_type: GuardrailType
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PIIEntity:
    """
    A detected PII entity.
    """

    entity_type: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class PIIAnalysisResult:
    """
    Result of PII analysis.
    """

    detected: bool
    entities: list[PIIEntity] = field(default_factory=list)
    action: GuardrailAction = GuardrailAction.ALLOW
    metadata: dict[str, Any] = field(default_factory=dict)