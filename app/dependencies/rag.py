from app.config.settings import settings

from app.providers.embedding.base import EmbeddingProvider
from app.providers.embedding.microsoft_foundry import (
    MicrosoftFoundryEmbeddingProvider,
)

from app.providers.vector_store.base import VectorStore
from app.providers.vector_store.qdrant import (
    QdrantVectorStore,
)

from app.rag.chunking.base import DocumentChunker
from app.rag.chunking.document_chunker import (
    HybridDocumentChunker,
)

from app.tokenization.tiktoken_counter import (
    TiktokenCounter,
)


# ============================================================
# SHARED TOKENIZER
# ============================================================

# The tokenizer corresponds to the embedding MODEL,
# not the deployment name.
_token_counter = TiktokenCounter(
    model_name=settings.foundry_embedding_model_name,
)


# ============================================================
# CHUNKER
# ============================================================

def get_document_chunker() -> DocumentChunker:
    """
    Create the configured document chunker.
    """

    match settings.chunking_strategy:

        case "hybrid":
            return HybridDocumentChunker(
                max_tokens=settings.chunk_max_tokens,
                token_counter=_token_counter.count,
            )

        case _:
            raise ValueError(
                f"Unsupported chunking strategy: "
                f"{settings.chunking_strategy}"
            )


# ============================================================
# EMBEDDING PROVIDER
# ============================================================

def get_embedding_provider() -> EmbeddingProvider:
    """
    Create the configured embedding provider.
    """

    match settings.embedding_provider:

        case "microsoft_foundry":
            return MicrosoftFoundryEmbeddingProvider(
                token_counter=_token_counter.count,
            )

        case _:
            raise ValueError(
                f"Unsupported embedding provider: "
                f"{settings.embedding_provider}"
            )


# ============================================================
# VECTOR STORE
# ============================================================

def get_vector_store() -> VectorStore:
    """
    Create the configured vector store.

    The application receives the VectorStore abstraction;
    the concrete Qdrant implementation is selected here.
    """

    return QdrantVectorStore()