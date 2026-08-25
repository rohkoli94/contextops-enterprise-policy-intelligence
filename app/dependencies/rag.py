from app.config.settings import settings

from app.providers.embedding.base import EmbeddingProvider
from app.providers.embedding.microsoft_foundry import (
    MicrosoftFoundryEmbeddingProvider,
)

from app.rag.chunking.base import DocumentChunker
from app.rag.chunking.document_chunker import (
    HybridDocumentChunker,
)

from app.tokenization.tiktoken_counter import (
    TiktokenCounter,
)


_token_counter = TiktokenCounter(
    model_name=settings.foundry_embedding_model_name,
)


def get_document_chunker() -> DocumentChunker:

    match settings.chunking_strategy:

        case "hybrid":
            return HybridDocumentChunker(
                max_tokens=settings.chunk_max_tokens, ## This is a tunable retrieval parameter, NOT the tokenizer limit.
                token_counter=_token_counter.count,
            )

        # Later:
        # case "semantic":
        #     return SemanticDocumentChunker(
        #         max_tokens=settings.chunk_max_tokens, ## This is a tunable retrieval parameter, NOT the tokenizer limit.
        #         token_counter=_token_counter.count,
        #     )

        case _:
            raise ValueError(
                f"Unsupported chunking strategy: "
                f"{settings.chunking_strategy}"
            )


def get_embedding_provider() -> EmbeddingProvider:

    match settings.embedding_provider:

        case "microsoft_foundry":
            return MicrosoftFoundryEmbeddingProvider(
                token_counter=_token_counter.count,
            )

        # Later:
        # case "openai":
        #     return OpenAIEmbeddingProvider(
        #         token_counter=_token_counter.count,
        #     )

        case _:
            raise ValueError(
                f"Unsupported embedding provider: "
                f"{settings.embedding_provider}"
            )