from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )

    gemini_model: str = Field(
        default="gemini-3.8-flash",
        validation_alias="GEMINI_MODEL",
    )

    explanation_model: str = Field(
        default="MBZUAI/LaMini-Flan-T5-783M",
        validation_alias="EXPLANATION_MODEL",
    )

    explanation_allow_gemini_fallback: bool = Field(
        default=True,
        validation_alias="EXPLANATION_ALLOW_GEMINI_FALLBACK",
    )

    host: str = Field(
        default="127.0.0.1",
        validation_alias="HOST",
    )

    port: int = Field(
        default=8000,
        validation_alias="PORT",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()