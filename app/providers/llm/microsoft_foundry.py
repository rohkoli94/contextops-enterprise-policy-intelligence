from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from app.config.settings import settings
from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse


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