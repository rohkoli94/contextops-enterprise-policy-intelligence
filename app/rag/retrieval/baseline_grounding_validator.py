import re

from langchain_core.documents import Document

from app.rag.retrieval.grounding_validator import (
    GroundingEvaluation,
    GroundingValidator,
)


class BaselineGroundingValidator(GroundingValidator):
    """
    Conservative Day 18 grounding validator.

    Current behavior:
        - validates that an answer exists
        - validates that evidence exists
        - extracts [SOURCE N] references
        - verifies that referenced sources exist

    This is an orchestration baseline.

    A stronger enterprise grounding implementation will later
    perform claim-level evidence verification.
    """

    _SOURCE_PATTERN = re.compile(
        r"\[SOURCE\s+(\d+)\]"
    )

    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        documents: list[Document],
    ) -> GroundingEvaluation:

        # --------------------------------------------------
        # QUERY VALIDATION
        # --------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # --------------------------------------------------
        # ANSWER VALIDATION
        # --------------------------------------------------

        if not answer or not answer.strip():
            return GroundingEvaluation(
                grounded=False,
                confidence="low",
                reason="Generated answer is empty.",
                supported_sources=[],
            )

        # --------------------------------------------------
        # EVIDENCE VALIDATION
        # --------------------------------------------------

        if not documents:
            return GroundingEvaluation(
                grounded=False,
                confidence="low",
                reason=(
                    "No evidence is available to ground "
                    "the answer."
                ),
                supported_sources=[],
            )

        # --------------------------------------------------
        # EXTRACT SOURCE REFERENCES
        # --------------------------------------------------

        source_numbers = [
            int(match.group(1))
            for match in self._SOURCE_PATTERN.finditer(
                answer
            )
        ]

        # --------------------------------------------------
        # VALID SOURCE REFERENCES
        # --------------------------------------------------

        valid_source_numbers = sorted(
            {
                number
                for number in source_numbers
                if 1 <= number <= len(documents)
            }
        )

        # --------------------------------------------------
        # NO USABLE SOURCE REFERENCES
        # --------------------------------------------------

        if not valid_source_numbers:
            return GroundingEvaluation(
                grounded=False,
                confidence="low",
                reason=(
                    "Answer does not reference the "
                    "available evidence."
                ),
                supported_sources=[],
            )

        # --------------------------------------------------
        # BASELINE SUCCESS
        # --------------------------------------------------

        return GroundingEvaluation(
            grounded=True,
            confidence="medium",
            reason=(
                "Answer references available evidence "
                "sources."
            ),
            supported_sources=valid_source_numbers,
        )