from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import (
    AIProjectClient as AsyncAIProjectClient,
)

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

    - Synchronous single embedding generation
    - Asynchronous single embedding generation
    - Synchronous batch embedding generation

    Batch limits enforced by this provider:

    - Maximum 2,048 inputs per API request.
    - Maximum 300,000 aggregate input tokens per API request.
    - Maximum 8,192 tokens for one input.

    The provider also validates that the number of returned
    embeddings matches the number of submitted inputs.
    """

    # ========================================================
    # API LIMITS
    # ========================================================

    MAX_INPUTS_PER_REQUEST = 2048

    MAX_TOKENS_PER_REQUEST = 300_000

    MAX_TOKENS_PER_INPUT = 8_192

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        token_counter,
    ) -> None:
        """
        Create the Microsoft Foundry embedding provider.

        token_counter:
            Callable used to count tokens using the tokenizer
            associated with the configured embedding model.

        Important:

            foundry_embedding_model_name
                is used for tokenizer selection and
                logical model identification.

            foundry_embedding_deployment_name
                is used for the actual Foundry API call.
        """

        # ----------------------------------------------------
        # SYNCHRONOUS CLIENT
        # ----------------------------------------------------

        self.credential = (
            DefaultAzureCredential()
        )

        self.project_client = (
            AIProjectClient(
                endpoint=(
                    settings.foundry_project_endpoint
                ),
                credential=self.credential,
            )
        )

        self.openai_client = (
            self.project_client.get_openai_client()
        )

        # ----------------------------------------------------
        # ASYNCHRONOUS CLIENT
        # ----------------------------------------------------

        self.async_credential = (
            AsyncDefaultAzureCredential()
        )

        self.async_project_client = (
            AsyncAIProjectClient(
                endpoint=(
                    settings.foundry_project_endpoint
                ),
                credential=self.async_credential,
            )
        )

        self.async_openai_client = (
            self.async_project_client.get_openai_client()
        )

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        # Inject the tokenizer so token counting remains
        # consistent with the configured embedding model.
        self.token_counter = token_counter

    # ========================================================
    # SYNCHRONOUS SINGLE EMBEDDING
    # ========================================================

    def generate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """
        Generate an embedding for a single text input.
        """

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        self._validate_single_input(
            request.text
        )

        # ----------------------------------------------------
        # FOUNDRY REQUEST
        # ----------------------------------------------------

        response = (
            self.openai_client.embeddings.create(
                # Actual Foundry deployment name.
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=request.text,
            )
        )

        # ----------------------------------------------------
        # RESPONSE VALIDATION
        # ----------------------------------------------------

        if len(response.data) != 1:
            raise ValueError(
                "Embedding response count does not match "
                "single input count."
            )

        return EmbeddingResponse(
            vector=response.data[0].embedding,

            # Logical model name.
            model=(
                settings.foundry_embedding_model_name
            ),

            provider="microsoft_foundry",
        )

    # ========================================================
    # ASYNCHRONOUS SINGLE EMBEDDING
    # ========================================================

    async def agenerate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """
        Generate an embedding asynchronously for a single
        text input.

        This method is used by the asynchronous query and
        retrieval pipeline.
        """

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        self._validate_single_input(
            request.text
        )

        # ----------------------------------------------------
        # FOUNDRY ASYNC REQUEST
        # ----------------------------------------------------

        response = (
            await self.async_openai_client
            .embeddings.create(
                # Actual Foundry deployment name.
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=request.text,
            )
        )

        # ----------------------------------------------------
        # RESPONSE VALIDATION
        # ----------------------------------------------------

        if len(response.data) != 1:
            raise ValueError(
                "Embedding response count does not match "
                "single input count."
            )

        return EmbeddingResponse(
            vector=response.data[0].embedding,

            # Logical model name.
            model=(
                settings.foundry_embedding_model_name
            ),

            provider="microsoft_foundry",
        )

    # ========================================================
    # SYNCHRONOUS BATCH EMBEDDING
    # ========================================================

    def generate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        """
        Generate embeddings for multiple text inputs.

        Large input lists are automatically divided into
        safe API batches based on:

        1. Maximum number of inputs.
        2. Maximum aggregate token count.
        3. Maximum tokens for one individual input.
        """

        # Nothing to embed.
        if not request.texts:
            return EmbeddingBatchResponse(
                vectors=[],
                model=(
                    settings.foundry_embedding_model_name
                ),
                provider="microsoft_foundry",
            )

        # ----------------------------------------------------
        # BUILD API-SAFE BATCHES
        # ----------------------------------------------------

        batches = self._build_safe_batches(
            request.texts
        )

        all_vectors: list[list[float]] = []

        # ----------------------------------------------------
        # PROCESS BATCHES
        # ----------------------------------------------------

        for batch in batches:

            # Send one safe batch to Microsoft Foundry.
            response = (
                self._generate_batch_request(
                    batch
                )
            )

            # ------------------------------------------------
            # PER-BATCH VALIDATION
            # ------------------------------------------------

            if len(response.data) != len(batch):
                raise ValueError(
                    "Embedding response count does not "
                    "match input count for batch."
                )

            # ------------------------------------------------
            # PRESERVE INPUT ORDER
            # ------------------------------------------------

            ordered_data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            all_vectors.extend(
                item.embedding
                for item in ordered_data
            )

        # ------------------------------------------------------
        # FINAL GLOBAL VALIDATION
        # ------------------------------------------------------

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

    # ========================================================
    # SAFE BATCH BUILDING
    # ========================================================

    def _build_safe_batches(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        """
        Split text inputs into API-safe batches.

        A new batch is started when adding another input would
        violate either:

        - MAX_INPUTS_PER_REQUEST
        - MAX_TOKENS_PER_REQUEST

        Every individual input is also validated against:

        - MAX_TOKENS_PER_INPUT
        """

        batches: list[list[str]] = []

        current_batch: list[str] = []

        current_token_count = 0

        for text in texts:

            # ------------------------------------------------
            # TOKEN COUNT
            # ------------------------------------------------

            token_count = self.token_counter(
                text
            )

            # ------------------------------------------------
            # SINGLE INPUT VALIDATION
            # ------------------------------------------------

            if (
                token_count
                > self.MAX_TOKENS_PER_INPUT
            ):
                raise ValueError(
                    "Embedding input exceeds the maximum "
                    f"allowed size of "
                    f"{self.MAX_TOKENS_PER_INPUT} tokens."
                )

            # ------------------------------------------------
            # INPUT LIMIT CHECK
            # ------------------------------------------------

            would_exceed_input_limit = (
                len(current_batch)
                >= self.MAX_INPUTS_PER_REQUEST
            )

            # ------------------------------------------------
            # TOKEN LIMIT CHECK
            # ------------------------------------------------

            would_exceed_token_limit = (
                current_token_count
                + token_count
                > self.MAX_TOKENS_PER_REQUEST
            )

            # ------------------------------------------------
            # START NEW BATCH
            # ------------------------------------------------

            if (
                current_batch
                and (
                    would_exceed_input_limit
                    or would_exceed_token_limit
                )
            ):
                batches.append(
                    current_batch
                )

                current_batch = []

                current_token_count = 0

            # ------------------------------------------------
            # ADD INPUT
            # ------------------------------------------------

            current_batch.append(
                text
            )

            current_token_count += (
                token_count
            )

        # ----------------------------------------------------
        # FINAL BATCH
        # ----------------------------------------------------

        if current_batch:
            batches.append(
                current_batch
            )

        return batches

    # ========================================================
    # BATCH API REQUEST
    # ========================================================

    def _generate_batch_request(
        self,
        texts: list[str],
    ):
        """
        Send one already-validated batch to
        Microsoft Foundry.
        """

        return (
            self.openai_client.embeddings.create(
                # Actual Foundry deployment name.
                model=(
                    settings.foundry_embedding_deployment_name
                ),
                input=texts,
            )
        )

    # ========================================================
    # SINGLE INPUT VALIDATION
    # ========================================================

    def _validate_single_input(
        self,
        text: str,
    ) -> None:
        """
        Validate one embedding input.
        """

        if not text or not text.strip():
            raise ValueError(
                "Embedding input cannot be empty."
            )

        # Token count uses the tokenizer corresponding
        # to the embedding MODEL name.
        token_count = self.token_counter(
            text
        )

        if (
            token_count
            > self.MAX_TOKENS_PER_INPUT
        ):
            raise ValueError(
                "Embedding input exceeds the maximum "
                f"allowed size of "
                f"{self.MAX_TOKENS_PER_INPUT} tokens."
            )

    # ========================================================
    # VECTOR DIMENSION
    # ========================================================

    def get_dimension(self) -> int:
        """
        Return the embedding vector dimension for the
        configured embedding model.
        """

        if settings.foundry_embedding_model_name == (
            "text-embedding-3-small"
        ):
            return 1536

        if settings.foundry_embedding_model_name == (
            "text-embedding-3-large"
        ):
            return 3072

        if settings.foundry_embedding_model_name == (
            "text-embedding-ada-002"
        ):
            return 1536

        raise ValueError(
            "Unsupported embedding model: "
            f"{settings.foundry_embedding_model_name}"
        )


# ============================================================
# ROHIT NOTES
# ============================================================

# Model name vs deployment name
#
# We deliberately keep these separate:
#
# FOUNDRY_EMBEDDING_MODEL_NAME
#     = text-embedding-3-small
#
# Used for:
#     tokenizer selection
#     logical model identification
#
#
# FOUNDRY_EMBEDDING_DEPLOYMENT_NAME
#     = contextops-embedding
#
# Used for:
#     Microsoft Foundry API request
#
#
# Flow:
#
#                 Settings
#                    |
#          +---------+---------+
#          |                   |
#          ↓                   ↓
#     Model name        Deployment name
#          |                   |
#          ↓                   ↓
#     TiktokenCounter     Foundry API
#
#
# This is important because an Azure / Foundry deployment name
# can be different from the underlying model name.
#
#
# ============================================================
# ASYNC QUERY PATH
# ============================================================
#
# Query
#   ↓
# EmbeddingProvider.agenerate()
#   ↓
# MicrosoftFoundryEmbeddingProvider.agenerate()
#   ↓
# AsyncOpenAI
#   ↓
# Microsoft Foundry
#   ↓
# EmbeddingResponse