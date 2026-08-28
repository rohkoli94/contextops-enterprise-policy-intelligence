from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

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
    - Single embedding generation
    - Batch embedding generation

    Batch limits enforced by this provider:

    - Maximum 2,048 inputs per API request.
    - Maximum 300,000 aggregate input tokens per API request.
    - Maximum 8,192 tokens for one input.

    The provider also validates that the number of returned
    embeddings matches the number of submitted inputs.
    """

    # Microsoft Foundry / Azure OpenAI embedding API limits.
    MAX_INPUTS_PER_REQUEST = 2048
    MAX_TOKENS_PER_REQUEST = 300_000
    MAX_TOKENS_PER_INPUT = 8_192

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
            foundry_embedding_model_name is used for tokenizer
            selection.

            foundry_embedding_deployment_name is used when
            calling the Microsoft Foundry embedding API.
        """

        self.credential = DefaultAzureCredential()

        self.project_client = AIProjectClient(
            endpoint=settings.foundry_project_endpoint,
            credential=self.credential,
        )

        self.openai_client = (
            self.project_client.get_openai_client()
        )

        # Inject the tokenizer so token counting is consistent
        # with the configured embedding model.
        self.token_counter = token_counter

    def generate(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """
        Generate an embedding for a single text input.
        """

        # Validate input before calling the provider.
        self._validate_single_input(
            request.text
        )

        response = self.openai_client.embeddings.create(
            # Deployment name is used for the actual Foundry API call.
            model=settings.foundry_embedding_deployment_name,
            input=request.text,
        )

        # A single input must produce exactly one embedding.
        if len(response.data) != 1:
            raise ValueError(
                "Embedding response count does not match "
                "single input count"
            )

        return EmbeddingResponse(
            vector=response.data[0].embedding,

            # Return the configured model name as the logical
            # embedding model, not the deployment identifier.
            model=settings.foundry_embedding_model_name,

            provider="microsoft_foundry",
        )

    def generate_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResponse:
        """
        Generate embeddings for multiple text inputs.

        Large input lists are automatically divided into safe
        API batches based on:

        1. Maximum number of inputs.
        2. Maximum aggregate token count.
        3. Maximum tokens for one individual input.
        """

        # Nothing to embed.
        if not request.texts:
            return EmbeddingBatchResponse(
                vectors=[],
                model=settings.foundry_embedding_model_name,
                provider="microsoft_foundry",
            )

        # Build API-safe batches.
        batches = self._build_safe_batches(
            request.texts
        )

        all_vectors: list[list[float]] = []

        for batch in batches:

            # Send one safe batch to Microsoft Foundry.
            response = self._generate_batch_request(
                batch
            )

            # --------------------------------------------------
            # PER-BATCH VALIDATION
            # --------------------------------------------------

            # Number of returned embeddings must equal the
            # number of submitted inputs for this batch.
            if len(response.data) != len(batch):
                raise ValueError(
                    "Embedding response count does not match "
                    "input count for batch"
                )

            # The API returns an index for each embedding.
            # Sort by index so the vector order matches the
            # original input order.
            ordered_data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            # Keep vectors as:
            #
            # list[list[float]]
            #
            # Example:
            #
            # [
            #     [0.1, 0.2, ...],
            #     [0.4, 0.8, ...],
            # ]
            all_vectors.extend(
                item.embedding
                for item in ordered_data
            )

        # ------------------------------------------------------
        # FINAL GLOBAL VALIDATION
        # ------------------------------------------------------

        # After combining all batches:
        #
        # total inputs == total vectors
        #
        # This guarantees that the one-to-one relationship
        # between DocumentChunks and embeddings is preserved.
        if len(all_vectors) != len(request.texts):
            raise ValueError(
                "Total embedding count does not match "
                "total input count"
            )

        return EmbeddingBatchResponse(
            vectors=all_vectors,
            model=settings.foundry_embedding_model_name,
            provider="microsoft_foundry",
        )

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

            # Count tokens using the tokenizer associated with
            # the configured embedding model.
            token_count = self.token_counter(text)

            # --------------------------------------------------
            # SINGLE INPUT VALIDATION
            # --------------------------------------------------

            if token_count > self.MAX_TOKENS_PER_INPUT:
                raise ValueError(
                    "Embedding input exceeds the maximum "
                    f"allowed size of "
                    f"{self.MAX_TOKENS_PER_INPUT} tokens."
                )

            # --------------------------------------------------
            # BATCH LIMIT CHECKS
            # --------------------------------------------------

            # This check happens BEFORE adding the next input.
            #
            # Example:
            #
            # MAX = 2048
            # current = 2048
            #
            # 2048 >= 2048 -> True
            #
            # Therefore the next input starts a new batch.
            would_exceed_input_limit = (
                len(current_batch)
                >= self.MAX_INPUTS_PER_REQUEST
            )

            # Exact equality is valid, so use >.
            #
            # Example:
            #
            # MAX = 300000
            # current + next = 300000
            #
            # 300000 > 300000 -> False
            would_exceed_token_limit = (
                current_token_count
                + token_count
                > self.MAX_TOKENS_PER_REQUEST
            )

            # --------------------------------------------------
            # START A NEW BATCH IF REQUIRED
            # --------------------------------------------------

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

            # Add current input to the active batch.
            current_batch.append(text)

            current_token_count += token_count

        # Add final batch.
        if current_batch:
            batches.append(
                current_batch
            )

        return batches

    def _generate_batch_request(
        self,
        texts: list[str],
    ):
        """
        Send one already-validated batch to Microsoft Foundry.
        """

        return self.openai_client.embeddings.create(
            # IMPORTANT:
            # Foundry API expects the deployment name here.
            model=settings.foundry_embedding_deployment_name,
            input=texts,
        )

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

        # Token count uses the tokenizer corresponding to the
        # embedding MODEL name.
        token_count = self.token_counter(text)

        if token_count > self.MAX_TOKENS_PER_INPUT:
            raise ValueError(
                "Embedding input exceeds the maximum "
                f"allowed size of "
                f"{self.MAX_TOKENS_PER_INPUT} tokens."
            )

    def get_dimension(self) -> int:
        """
        Return the embedding vector dimension for the configured
        embedding model.
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