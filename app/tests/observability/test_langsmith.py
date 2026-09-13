import os

from app.observability.langsmith import (
    configure_langsmith,
)


def test_langsmith_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_tracing",
        False,
    )

    assert (
        configure_langsmith()
        is False
    )


def test_langsmith_requires_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_tracing",
        True,
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_api_key",
        None,
    )

    assert (
        configure_langsmith()
        is False
    )


def test_langsmith_sets_environment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_tracing",
        True,
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_api_key",
        "test-key",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_project",
        "contextops-test",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_endpoint",
        "https://example.com",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_workspace_id",
        "workspace-001",
    )

    assert (
        configure_langsmith()
        is True
    )

    assert os.environ[
        "LANGSMITH_TRACING"
    ] == "true"

    assert os.environ[
        "LANGSMITH_API_KEY"
    ] == "test-key"

    assert os.environ[
        "LANGSMITH_PROJECT"
    ] == "contextops-test"

    assert os.environ[
        "LANGSMITH_ENDPOINT"
    ] == "https://example.com"

    assert os.environ[
        "LANGSMITH_WORKSPACE_ID"
    ] == "workspace-001"


def test_langsmith_removes_stale_workspace_id(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LANGSMITH_WORKSPACE_ID",
        "stale-workspace",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_tracing",
        True,
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_api_key",
        "test-key",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_project",
        "contextops-test",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_endpoint",
        "https://example.com",
    )

    monkeypatch.setattr(
        "app.observability.langsmith.settings.langsmith_workspace_id",
        None,
    )

    assert (
        configure_langsmith()
        is True
    )

    assert (
        "LANGSMITH_WORKSPACE_ID"
        not in os.environ
    )