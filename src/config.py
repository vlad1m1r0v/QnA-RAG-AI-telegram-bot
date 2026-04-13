from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
    PydanticBaseSettingsSource
)


class LLMConfig(BaseModel):
    model: str
    temperature: float


class EmbeddingsConfig(BaseModel):
    model: str
    device: str
    vector_size: int
    embed_batch_size: int


class QdrantConfig(BaseModel):
    top_k: int
    distance: str


class ScraperConfig(BaseModel):
    user_agent: str
    requests_per_second: int


class TextSplitterConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    separators: list[str]


class MemoryConfig(BaseModel):
    window_size: int
    summary_max_sentences: int


class AppConfig(BaseSettings):
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    qdrant: QdrantConfig
    scraper: ScraperConfig
    text_splitter: TextSplitterConfig
    memory: MemoryConfig

    model_config = SettingsConfigDict(toml_file="config.toml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


config = AppConfig()
