import pytest

from app.guardrails.pii import RegexPIIAnalyzer


@pytest.mark.asyncio
async def test_pii_detects_email() -> None:
    analyzer = RegexPIIAnalyzer()

    result = await analyzer.analyze(
        "Please contact john.doe@example.com.",
    )

    assert result.detected is True
    assert any(
        entity.entity_type == "email"
        for entity in result.entities
    )


@pytest.mark.asyncio
async def test_pii_detects_indian_phone_number() -> None:
    analyzer = RegexPIIAnalyzer()

    result = await analyzer.analyze(
        "Call me at +919876543210.",
    )

    assert result.detected is True
    assert any(
        entity.entity_type == "phone"
        for entity in result.entities
    )


@pytest.mark.asyncio
async def test_pii_allows_normal_text() -> None:
    analyzer = RegexPIIAnalyzer()

    result = await analyzer.analyze(
        "What is the leave policy?",
    )

    assert result.detected is False
    assert result.entities == []