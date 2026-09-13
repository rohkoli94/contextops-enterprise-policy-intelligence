import logging
import os

from app.config.settings import settings


logger = logging.getLogger("contextops.langsmith")


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing for ContextOps.

    LangGraph/LangChain automatically uses the standard
    LANGSMITH_* environment variables for tracing.

    Returns:
        True when tracing is enabled and configured.
        False otherwise.
    """

    if not settings.langsmith_tracing:
        logger.info(
            "LangSmith tracing is disabled."
        )
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LangSmith tracing is enabled but "
            "LANGSMITH_API_KEY is not configured."
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"

    os.environ["LANGSMITH_API_KEY"] = (
        settings.langsmith_api_key
    )

    os.environ["LANGSMITH_PROJECT"] = (
        settings.langsmith_project
    )

    os.environ["LANGSMITH_ENDPOINT"] = (
        settings.langsmith_endpoint
    )

    # Workspace ID is only necessary for configurations where
    # the API key has access to multiple workspaces.
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = (
            settings.langsmith_workspace_id
        )
    else:
        os.environ.pop(
            "LANGSMITH_WORKSPACE_ID",
            None,
        )

    logger.info(
        "LangSmith tracing configured.",
        extra={
            "project": settings.langsmith_project,
            "endpoint": settings.langsmith_endpoint,
            "workspace_configured": bool(
                settings.langsmith_workspace_id
            ),
        },
    )

    return True