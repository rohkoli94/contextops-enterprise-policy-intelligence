from app.config.settings import settings

from app.providers.embedding.base import EmbeddingProvider
from app.providers.embedding.microsoft_foundry import (
    MicrosoftFoundryEmbeddingProvider,
)
from app.providers.embedding.bm25 import (
    BM25SparseEmbeddingProvider,
)
from app.providers.embedding.sparse_base import (
    SparseEmbeddingProvider,
)

from app.providers.vector_store.base import VectorStore
from app.providers.vector_store.qdrant import (
    QdrantVectorStore,
)

from app.rag.chunking.base import DocumentChunker
from app.rag.chunking.document_chunker import (
    HybridDocumentChunker,
)

from app.rag.retrieval.hybrid import (
    HybridRetriever,
)

from app.tokenization.tiktoken_counter import (
    TiktokenCounter,
)


# ============================================================
# SHARED TOKENIZER
# ============================================================

_token_counter = TiktokenCounter(
    model_name=settings.foundry_embedding_model_name,
)


# ============================================================
# SHARED DENSE EMBEDDING PROVIDER
# ============================================================

_embedding_provider = None


def get_embedding_provider() -> EmbeddingProvider:
    """
    Return the configured dense embedding provider.

    The provider is created once and reused by the application.
    """

    global _embedding_provider

    if _embedding_provider is not None:
        return _embedding_provider

    match settings.embedding_provider:

        case "microsoft_foundry":
            _embedding_provider = (
                MicrosoftFoundryEmbeddingProvider(
                    token_counter=_token_counter.count,
                )
            )

        case _:
            raise ValueError(
                f"Unsupported embedding provider: "
                f"{settings.embedding_provider}"
            )

    return _embedding_provider


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
# SHARED VECTOR STORE
# ============================================================

_vector_store = QdrantVectorStore()


def get_vector_store() -> VectorStore:
    """
    Return the configured vector store.

    Collection initialization is intentionally NOT performed
    here. Qdrant infrastructure is initialized once during
    application startup.
    """

    return _vector_store


# ============================================================
# SHARED SPARSE EMBEDDING PROVIDER
# ============================================================

_bm25_provider = BM25SparseEmbeddingProvider()


def get_sparse_embedding_provider() -> SparseEmbeddingProvider:
    """
    Return the configured sparse embedding provider.
    """

    match settings.sparse_embedding_provider:

        case "bm25":
            return _bm25_provider

        # Later:
        # case "splade":
        #     return SPLADESparseEmbeddingProvider()

        case _:
            raise ValueError(
                f"Unsupported sparse embedding provider: "
                f"{settings.sparse_embedding_provider}"
            )


# ============================================================
# HYBRID RETRIEVER
# ============================================================

_hybrid_retriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """
    Return the configured hybrid retriever.

    The retriever reuses the shared dense provider,
    sparse provider, and vector store.
    """

    global _hybrid_retriever

    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            embedding_provider=get_embedding_provider(),
            sparse_embedding_provider=(
                get_sparse_embedding_provider()
            ),
            vector_store=get_vector_store(),
        )

    return _hybrid_retriever