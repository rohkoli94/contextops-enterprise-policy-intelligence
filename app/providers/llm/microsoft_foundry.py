import base64

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from app.config.settings import settings
from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse, VisionRequest


class MicrosoftFoundryProvider(LLMProvider):
    def __init__(self) -> None:
        self.credential = DefaultAzureCredential()

        self.project_client = AIProjectClient(
            endpoint=settings.foundry_project_endpoint,
            credential=self.credential,
        )

        self.openai_client = self.project_client.get_openai_client()

    def generate(self, request: LLMRequest) -> LLMResponse:
        input_text = request.user_prompt

        if request.context:
            input_text = (
                f"Context:\n{request.context}\n\n"
                f"Question:\n{request.user_prompt}"
            )

        response = self.openai_client.responses.create(
            model=settings.foundry_model_name,
            instructions=request.system_prompt,
            input=input_text,
        )

        return LLMResponse(
            content=response.output_text,
            model=settings.foundry_model_name,
            provider="microsoft_foundry",
        )


    def generate_vision(
        self,
        request: VisionRequest,
    ) -> LLMResponse:
        """
        Generate a response using a vision-capable model.

        The extracted image bytes are converted into a Base64
        data URL before being sent to Microsoft Foundry.
        """

        # Convert image bytes into Base64 text.
        encoded_image = base64.b64encode(
            request.image_bytes
        ).decode("utf-8")

        # Build a data URL understood as image input.
        image_data_url = (
            f"data:{request.media_type};base64,"
            f"{encoded_image}"
        )

        response = self.openai_client.responses.create(
            model=settings.foundry_vision_model_name,
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
            model=settings.foundry_model_name,
            provider="microsoft_foundry",
        )