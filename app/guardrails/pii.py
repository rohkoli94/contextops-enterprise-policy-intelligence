import re

from app.guardrails.base import PIIAnalyzer
from app.guardrails.models import (
    GuardrailAction,
    PIIAnalysisResult,
    PIIEntity,
)


class RegexPIIAnalyzer(PIIAnalyzer):
    """
    Baseline PII analyzer for common, obvious PII patterns.

    Current detection:
    - email
    - Indian phone number
    - Aadhaar-like 12 digit number

    The abstraction is intentionally provider-independent so
    it can later be replaced by a stronger enterprise DLP/PII
    implementation without changing the workflow.
    """

    _PATTERNS = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "phone": re.compile(
            r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)"
        ),
        "aadhaar": re.compile(
            r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"
        ),
    }

    async def analyze(
        self,
        text: str,
    ) -> PIIAnalysisResult:

        if not text:
            return PIIAnalysisResult(
                detected=False,
            )

        entities: list[PIIEntity] = []

        for entity_type, pattern in self._PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )

        if not entities:
            return PIIAnalysisResult(
                detected=False,
                action=GuardrailAction.ALLOW,
            )

        return PIIAnalysisResult(
            detected=True,
            entities=entities,
            action=GuardrailAction.AUDIT,
            metadata={
                "entity_count": len(entities),
                "entity_types": sorted(
                    {entity.entity_type for entity in entities}
                ),
            },
        )

# note
# For input PII, we're currently returning: detected → AUDIT
# rather than automatically blocking. That's deliberate. 
# In an enterprise policy system, some legitimate queries may contain an identifier that is necessary for the task. 
# The actual action should be controlled by policy rather than blindly blocking every PII occurrence.
