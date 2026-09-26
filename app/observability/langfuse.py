import logging
import os

from langfuse import get_client

from app.config.settings import settings


logger = logging.getLogger("contextops.langfuse")


def configure_langfuse() -> bool:
    """Configure Langfuse tracing for ContextOps."""

    if not settings.langfuse_tracing:
        logger.info("Langfuse tracing is disabled.")
        return False

    if not settings.langfuse_public_key:
        logger.warning(
            "Langfuse tracing is enabled but "
            "LANGFUSE_PUBLIC_KEY is not configured."
        )
        return False

    if not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse tracing is enabled but "
            "LANGFUSE_SECRET_KEY is not configured."
        )
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = (
        settings.langfuse_public_key
    )

    os.environ["LANGFUSE_SECRET_KEY"] = (
        settings.langfuse_secret_key
    )

    os.environ["LANGFUSE_BASE_URL"] = (
        settings.langfuse_base_url
    )

    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = (
        settings.langfuse_tracing_environment
    )

    logger.info(
        "Langfuse tracing configured.",
        extra={
            "base_url": settings.langfuse_base_url,
            "environment": (
                settings.langfuse_tracing_environment
            ),
        },
    )

    return True


def get_langfuse_client():
    """Return the configured Langfuse singleton client."""
    return get_client()