from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse


class QueryService:
    def __init__(self, llm_provider: LLMProvider) :
        self.llm_provider = llm_provider

    def ask(self, question: str) -> LLMResponse:
        request = LLMRequest(
            system_prompt="You are an entripse policy intelligence assistant." \
            "Please povide clear answer and donot hallunicate",
            user_prompt=question
        )
        return self.llm_provider.generate(request)