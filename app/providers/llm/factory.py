from app.config.settings import settings
from app.providers.llm.base import LLMProvider
from app.providers.llm.microsoft_foundry import (
    MicrosoftFoundryProvider,
)


def get_llm_provider() -> LLMProvider:
    """
    Create the configured LLM provider.

    The application depends on the LLMProvider abstraction,
    allowing different LLM implementations to be introduced
    without changing the query workflow.
    """

    match settings.llm_provider:

        case "microsoft_foundry":
            return MicrosoftFoundryProvider()

        case "open_source":
            raise ValueError(
                "The 'open_source' LLM provider is not implemented yet."
            )

        case _:
            raise ValueError(
                f"Unsupported LLM provider: "
                f"{settings.llm_provider}"
            )