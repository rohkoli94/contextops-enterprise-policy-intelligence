from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ContextOps"
    app_version: str = "0.1.0"
    environment: str = "development"

    # ========================================================
    # FastAPI
    # ========================================================

    api_base_url: str = "http://127.0.0.1:8000"

    # ========================================================
    # Microsoft Foundry
    # ========================================================

    foundry_project_endpoint: str

    # LLM
    foundry_model_name: str
    foundry_model_deployment_name: str

    # Vision
    foundry_vision_model_name: str
    foundry_vision_model_deployment_name: str

    # Embeddings
    #
    # Model name:
    # Used for tokenizer/model identification.
    foundry_embedding_model_name: str

    # Deployment name:
    # Used for the actual Foundry embedding API request.
    foundry_embedding_deployment_name: str

    # ========================================================
    # RAG Strategy
    # ========================================================

    embedding_provider: str = "microsoft_foundry"
    chunking_strategy: str = "hybrid"

    # Maximum token budget for a retrieval chunk.
    #
    # This is a configurable retrieval parameter.
    # The tokenizer measures actual token usage against this value.
    chunk_max_tokens: int = 800

    # ========================================================
    # Azure Blob Storage
    # ========================================================

    azure_storage_connection_string: str
    azure_storage_container_name: str = "contextops-documents"

    # ========================================================
    # PostgreSQL
    # ========================================================

    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()