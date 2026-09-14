from collections.abc import Callable
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
from azure.identity import DefaultAzureCredential
from azure.identity.aio import (
    DefaultAzureCredential as AsyncDefaultAzureCredential,
)

from app.config.settings import settings
from app.providers.embedding.base import (
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
)


class MicrosoftFoundryEmbeddingProvider(
    EmbeddingProvider
):
    """
    Microsoft Foundry embedding provider.

    Supports:

        - synchronous single embedding
        - asynchronous single embedding
        - synchronous batch embedding
        - asynchronous batch embedding

    Provider-specific request limits are handled inside this
    implementation so higher-level services remain provider agnostic.
    """

    # ============================================================
    # PROVIDER LIMITS
    # ============================================================

    MAX_INPUTS_PER_REQUEST = 2048
    MAX_TOKENS_PER_REQUEST = 300_000
    MAX_TOKENS_PER_INPUT = 8192

    def __init__(
        self,
        token_counter: Callable[[str], int],
    ) -> None:
        self.token_counter = token_counter

        # ========================================================
        # SYNCHRONOUS CLIENTS
        # ========================================================

        self.credential = DefaultAzureCredential()

        self.project_client = AIProjectClient(
            endpoint=settings.foundry_project_endpoint,
            credential=self.credential,
        )

        self.openai_client = (
            self.project_client.get_openai_client()
        )

        # ========================================================
        # ASYNCHRONOUS CLIENTS
        # ========================================================

        self.async_credential = (
            AsyncDefaultAzureCredential()
        )

        self.async_project_client = AsyncAIProjectClient(
            endpoint=settings.foundry_project_endpoint,
            credential=self.async_credential,
        )

        self.async_openai_client = (
            self.async_project_client.get_openai_client()
        )

    # ============================================================
    # SYNCHRONOUS SINGLE EMBEDDING
    # ============================================================

    def generate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        self._validate_single_input(
            request.text
        )

        response = (
            self.openai_client.embeddings.create(
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=request.text,
            )
        )

        if len(response.data) != 1:
            raise ValueError(
                "Embedding response count does not match "
                "single input count."
            )

        return EmbeddingResponse(
            vector=response.data[0].embedding,
            model=(
                settings.foundry_embedding_model_name
            ),
            provider="microsoft_foundry",
        )

    # ============================================================
    # ASYNCHRONOUS SINGLE EMBEDDING
    # ============================================================

    async def agenerate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        self._validate_single_input(
            request.text
        )

        response = (
            await self.async_openai_client
            .embeddings.create(
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=request.text,
            )
        )

        if len(response.data) != 1:
            raise ValueError(
                "Embedding response count does not match "
                "single input count."
            )

        return EmbeddingResponse(
            vector=response.data[0].embedding,
            model=(
                settings.foundry_embedding_model_name
            ),
            provider="microsoft_foundry",
        )

    # ============================================================
    # SYNCHRONOUS BATCH EMBEDDING
    # ============================================================

    def generate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        if not request.texts:
            return EmbeddingBatchResponse(
                vectors=[],
                model=(
                    settings.foundry_embedding_model_name
                ),
                provider="microsoft_foundry",
            )

        batches = self._build_safe_batches(
            request.texts
        )

        all_vectors: list[list[float]] = []

        for batch in batches:
            response = (
                self._generate_batch_request(
                    batch
                )
            )

            if len(response.data) != len(batch):
                raise ValueError(
                    "Embedding response count does not "
                    "match input count for batch."
                )

            # The API response contains an index for each input.
            # Sorting ensures the final vector order exactly matches
            # the original input order.
            ordered_data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            all_vectors.extend(
                item.embedding
                for item in ordered_data
            )

        if len(all_vectors) != len(request.texts):
            raise ValueError(
                "Total embedding count does not match "
                "total input count."
            )

        return EmbeddingBatchResponse(
            vectors=all_vectors,
            model=(
                settings.foundry_embedding_model_name
            ),
            provider="microsoft_foundry",
        )

    # ============================================================
    # ASYNCHRONOUS BATCH EMBEDDING
    # ============================================================

    async def agenerate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        """
        Generate embeddings asynchronously.

        Large input lists are divided into provider-safe batches.

        Each batch is sent through the native asynchronous
        Microsoft Foundry/OpenAI client.

        Batches are processed sequentially to keep provider-side
        request pressure controlled while each individual network
        request remains non-blocking.
        """

        if request is None:
            raise ValueError(
                "Embedding batch request cannot be None."
            )

        if not request.texts:
            return EmbeddingBatchResponse(
                vectors=[],
                model=(
                    settings.foundry_embedding_model_name
                ),
                provider="microsoft_foundry",
            )

        batches = self._build_safe_batches(
            request.texts
        )

        all_vectors: list[list[float]] = []

        for batch in batches:
            response = (
                await self._agenerate_batch_request(
                    batch
                )
            )

            if len(response.data) != len(batch):
                raise ValueError(
                    "Embedding response count does not "
                    "match input count for async batch."
                )

            ordered_data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            all_vectors.extend(
                item.embedding
                for item in ordered_data
            )

        if len(all_vectors) != len(request.texts):
            raise ValueError(
                "Total async embedding count does not "
                "match total input count."
            )

        return EmbeddingBatchResponse(
            vectors=all_vectors,
            model=(
                settings.foundry_embedding_model_name
            ),
            provider="microsoft_foundry",
        )

    # ============================================================
    # SAFE BATCH BUILDING
    # ============================================================

    def _build_safe_batches(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        """
        Split inputs according to provider constraints.

        Constraints:

            MAX_INPUTS_PER_REQUEST
            MAX_TOKENS_PER_REQUEST
            MAX_TOKENS_PER_INPUT
        """

        batches: list[list[str]] = []

        current_batch: list[str] = []
        current_token_count = 0

        for text in texts:
            self._validate_single_input(
                text
            )

            token_count = self.token_counter(
                text
            )

            would_exceed_input_limit = (
                len(current_batch)
                >= self.MAX_INPUTS_PER_REQUEST
            )

            would_exceed_token_limit = (
                current_token_count
                + token_count
                > self.MAX_TOKENS_PER_REQUEST
            )

            if current_batch and (
                would_exceed_input_limit
                or would_exceed_token_limit
            ):
                batches.append(
                    current_batch
                )

                current_batch = []
                current_token_count = 0

            current_batch.append(
                text
            )

            current_token_count += token_count

        if current_batch:
            batches.append(
                current_batch
            )

        return batches

    # ============================================================
    # SYNCHRONOUS BATCH REQUEST
    # ============================================================

    def _generate_batch_request(
        self,
        texts: list[str],
    ) -> Any:
        """
        Send one already-validated batch synchronously.
        """

        return (
            self.openai_client.embeddings.create(
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=texts,
            )
        )

    # ============================================================
    # ASYNCHRONOUS BATCH REQUEST
    # ============================================================

    async def _agenerate_batch_request(
        self,
        texts: list[str],
    ) -> Any:
        """
        Send one already-validated batch using the native
        asynchronous client.
        """

        return (
            await self.async_openai_client
            .embeddings.create(
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=texts,
            )
        )

    # ============================================================
    # SINGLE INPUT VALIDATION
    # ============================================================

    def _validate_single_input(
        self,
        text: str,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(
                "Embedding input must be a string."
            )

        if not text or not text.strip():
            raise ValueError(
                "Embedding input cannot be empty."
            )

        token_count = self.token_counter(
            text
        )

        if token_count > self.MAX_TOKENS_PER_INPUT:
            raise ValueError(
                "Embedding input exceeds the maximum "
                f"allowed size of "
                f"{self.MAX_TOKENS_PER_INPUT} tokens."
            )

    # ============================================================
    # VECTOR DIMENSION
    # ============================================================

    def get_dimension(self) -> int:
        """
        Return the dimensionality of the configured embedding
        model.

        The ContextOps configuration currently uses
        text-embedding-3-small.
        """

        return 1536

    # ============================================================
    # RESOURCE CLEANUP
    # ============================================================

    def close(self) -> None:
        """
        Close synchronous provider resources.
        """

        try:
            self.project_client.close()
        finally:
            self.credential.close()

    async def aclose(self) -> None:
        """
        Close asynchronous provider resources.
        """

        try:
            await self.async_project_client.close()
        finally:
            await self.async_credential.close()