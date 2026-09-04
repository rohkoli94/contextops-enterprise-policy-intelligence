import re

from app.guardrails.base import Guardrail
from app.guardrails.models import (
    GuardrailAction,
    GuardrailResult,
    GuardrailType,
)


class PromptInjectionGuard(Guardrail):
    """
    Baseline prompt-injection detector.

    This is intentionally deterministic for Day 18.

    The interface allows us to replace this later with:
    - classifier-based detection
    - dedicated security model
    - managed safety service
    - hybrid rule + ML detection

    The same abstraction can be used for both:
    - user input
    - retrieved document content
    """

    _PATTERNS = (
        r"\bignore\s+(all\s+)?previous\s+instructions\b",
        r"\bignore\s+(the\s+)?system\s+prompt\b",
        r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
        r"\bforget\s+(all\s+)?previous\s+instructions\b",
        r"\breveal\s+(the\s+)?system\s+prompt\b",
        r"\bshow\s+(me\s+)?(the\s+)?system\s+prompt\b",
        r"\breveal\s+(the\s+)?developer\s+message\b",
        r"\bshow\s+(me\s+)?(the\s+)?developer\s+message\b",
        r"\bjailbreak\b",
    )

    async def evaluate(
        self,
        *,
        text: str,
        **_: object,
    ) -> GuardrailResult:

        if not text or not text.strip():
            return GuardrailResult(
                allowed=True,
                action=GuardrailAction.ALLOW,
                guardrail_type=GuardrailType.PROMPT_INJECTION,
            )

        normalized_text = text.lower()

        for pattern in self._PATTERNS:
            if re.search(pattern, normalized_text):
                return GuardrailResult(
                    allowed=False,
                    action=GuardrailAction.BLOCK,
                    guardrail_type=GuardrailType.PROMPT_INJECTION,
                    reason="Potential prompt injection detected.",
                    metadata={
                        "pattern": pattern,
                    },
                )

        return GuardrailResult(
            allowed=True,
            action=GuardrailAction.ALLOW,
            guardrail_type=GuardrailType.PROMPT_INJECTION,
        )