from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ContextOps"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_base_url: str = "http://127.0.0.1:8080"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()