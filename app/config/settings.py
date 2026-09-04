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

    # ========================================================
    # Qdrant
    # ========================================================

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "contextops_documents"

    qdrant_sharding_method: str = "custom"
    qdrant_shard_number: int = 1
    qdrant_default_shard_key: str = "default"

    default_tenant_id: str = "contextops"

    # BM25 sparse embedding model
    bm25_model_name: str = "Qdrant/bm25"
    sparse_embedding_provider: str = "bm25"

    #FastEmbed - For production Docker, we'll later override it with something such as:/app/.cache/fastembed
    fastembed_cache_dir: str = ".cache/fastembed" 

    # ========================================================
    # Retrieval / Query
    # ========================================================

    retrieval_top_k: int = 10

    conversation_recent_message_limit: int = 6

    rerank_candidate_limit: int = 20

    # Maximum number of reranked documents that ContextOps
    # may place into the final context.
    context_max_documents: int = 5

    # Hard token budget for the final LLM context.
    context_max_tokens: int = 6000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()