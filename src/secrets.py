from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Secrets(BaseSettings):
    hf_token: str = Field(alias="HUGGING_FACE_ACCESS_TOKEN")

    qdrant_url: str = Field(alias="QDRANT_URL")
    qdrant_collection_name: str = Field(alias="QDRANT_COLLECTION_NAME")

    groq_api_key: str = Field(alias="GROQ_API_KEY")
    assembly_ai_api_key: str = Field(alias="ASSEMBLY_AI_API_KEY")
    telegram_bot_key: str = Field(alias="TELEGRAM_BOT_KEY")
    mongodb_url: str = Field(alias="MONGODB_URL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


secrets = Secrets()
