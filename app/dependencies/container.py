from app.dependencies.rag import (
    get_hybrid_retriever,
)
from app.providers.llm.microsoft_foundry import (
    MicrosoftFoundryProvider,
)
from app.services.query_service import (
    QueryService,
)


def create_query_service() -> QueryService:
    """
    Build the shared QueryService and its dependencies.

    This function is used during FastAPI application startup.

    Dependencies:

        MicrosoftFoundryProvider
                +
        HybridRetriever
                ↓
          QueryService
    """

    llm_provider = (
        MicrosoftFoundryProvider()
    )

    hybrid_retriever = (
        get_hybrid_retriever()
    )

    return QueryService(
        llm_provider=llm_provider,
        hybrid_retriever=hybrid_retriever,
    )