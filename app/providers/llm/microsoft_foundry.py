import base64

from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import (
    AIProjectClient as AsyncAIProjectClient,
)

from azure.identity import DefaultAzureCredential
from azure.identity.aio import (
    DefaultAzureCredential as AsyncDefaultAzureCredential,
)

from app.config.settings import settings
from app.providers.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    VisionRequest,
)


class MicrosoftFoundryProvider(LLMProvider):
    """
    Microsoft Foundry provider for:

    - normal LLM generation
    - vision generation
    - asynchronous query execution

    Synchronous clients are retained for existing synchronous
    application flows.

    Asynchronous clients are used by the production query path.
    """

    def __init__(self) -> None:
        # ====================================================
        # SYNCHRONOUS CLIENTS
        # ====================================================

        self.credential = DefaultAzureCredential()

        self.project_client = AIProjectClient(
            endpoint=settings.foundry_project_endpoint,
            credential=self.credential,
        )

        self.openai_client = (
            self.project_client.get_openai_client()
        )

        # ====================================================
        # ASYNCHRONOUS CLIENTS
        # ====================================================

        self.async_credential = (
            AsyncDefaultAzureCredential()
        )

        self.async_project_client = (
            AsyncAIProjectClient(
                endpoint=settings.foundry_project_endpoint,
                credential=self.async_credential,
            )
        )

        self.async_openai_client = (
            self.async_project_client.get_openai_client()
        )

    # ========================================================
    # SYNCHRONOUS TEXT GENERATION
    # ========================================================

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response using the configured LLM deployment.
        """

        input_text = request.user_prompt

        if request.context:
            input_text = (
                f"Context:\n{request.context}\n\n"
                f"Question:\n{request.user_prompt}"
            )

        response = self.openai_client.responses.create(
            model=(
                settings.foundry_model_deployment_name
            ),
            instructions=request.system_prompt,
            input=input_text,
        )

        return LLMResponse(
            content=response.output_text,
            model=settings.foundry_model_name,
            provider="microsoft_foundry",
        )

    # ========================================================
    # ASYNCHRONOUS TEXT GENERATION
    # ========================================================

    async def agenerate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response asynchronously using the
        configured Microsoft Foundry deployment.
        """

        input_text = request.user_prompt

        if request.context:
            input_text = (
                f"Context:\n{request.context}\n\n"
                f"Question:\n{request.user_prompt}"
            )

        response = (
            await self.async_openai_client
            .responses
            .create(
                model=(
                    settings.foundry_model_deployment_name
                ),
                instructions=request.system_prompt,
                input=input_text,
            )
        )

        return LLMResponse(
            content=response.output_text,
            model=settings.foundry_model_name,
            provider="microsoft_foundry",
        )

    # ========================================================
    # SYNCHRONOUS VISION GENERATION
    # ========================================================

    def generate_vision(
        self,
        request: VisionRequest,
    ) -> LLMResponse:
        """
        Generate a response using the configured vision
        deployment.

        The extracted image bytes are converted into a
        Base64 data URL and sent as image input.
        """

        encoded_image = base64.b64encode(
            request.image_bytes
        ).decode("utf-8")

        image_data_url = (
            f"data:{request.media_type};base64,"
            f"{encoded_image}"
        )

        response = self.openai_client.responses.create(
            model=(
                settings.foundry_vision_model_deployment_name
            ),
            instructions=request.system_prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": request.user_prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                        },
                    ],
                },
            ],
        )

        return LLMResponse(
            content=response.output_text,
            model=settings.foundry_vision_model_name,
            provider="microsoft_foundry",
        )

    # ========================================================
    # ASYNCHRONOUS VISION GENERATION
    # ========================================================

    async def agenerate_vision(
        self,
        request: VisionRequest,
    ) -> LLMResponse:
        """
        Generate a response asynchronously using the
        configured vision deployment.
        """

        encoded_image = base64.b64encode(
            request.image_bytes
        ).decode("utf-8")

        image_data_url = (
            f"data:{request.media_type};base64,"
            f"{encoded_image}"
        )

        response = (
            await self.async_openai_client
            .responses
            .create(
                model=(
                    settings.foundry_vision_model_deployment_name
                ),
                instructions=request.system_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": request.user_prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": image_data_url,
                            },
                        ],
                    },
                ],
            )
        )

        return LLMResponse(
            content=response.output_text,
            model=settings.foundry_vision_model_name,
            provider="microsoft_foundry",
        )

    # ========================================================
    # ASYNC CLIENT LIFECYCLE
    # ========================================================

    async def aclose(self) -> None:
        """
        Close asynchronous Microsoft Foundry resources.

        Called during FastAPI application shutdown.
        """

        await self.async_project_client.close()
        await self.async_credential.close()