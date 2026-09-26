import os

from app.observability.langfuse import configure_langfuse


def test_langfuse_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_tracing",
        False,
    )
    assert configure_langfuse() is False


def test_langfuse_requires_public_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_tracing",
        True,
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_public_key",
        None,
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_secret_key",
        "test-secret",
    )
    assert configure_langfuse() is False


def test_langfuse_requires_secret_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_tracing",
        True,
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_public_key",
        "test-public",
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_secret_key",
        None,
    )
    assert configure_langfuse() is False


def test_langfuse_sets_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_tracing",
        True,
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_public_key",
        "test-public",
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_secret_key",
        "test-secret",
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_base_url",
        "https://example.com",
    )
    monkeypatch.setattr(
        "app.observability.langfuse.settings.langfuse_tracing_environment",
        "test",
    )

    assert configure_langfuse() is True
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "test-public"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "test-secret"
    assert os.environ["LANGFUSE_BASE_URL"] == "https://example.com"
    assert os.environ["LANGFUSE_TRACING_ENVIRONMENT"] == "test"
