from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    hf_token: str = Field(alias="HUGGING_FACE_ACCESS_TOKEN")
    embedding_model: str = Field(alias="EMBEDDING_MODEL")

    qdrant_url: str = Field(alias="QDRANT_URL")
    collection_name: str = Field(alias="QDRANT_COLLECTION_NAME")

    user_agent: str = Field(alias="USER_AGENT")

    groq_api_key: str = Field(alias="GROQ_API_KEY")
    llm_model: str = Field(alias="LLM_MODEL")
    top_k: int = 6

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()