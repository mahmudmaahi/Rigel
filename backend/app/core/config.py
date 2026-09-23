from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Rigel API"
    service_name: str = "rigel-api"
    version: str = "0.1.0"
    max_audio_upload_bytes: int = 100 * 1024 * 1024
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(env_prefix="RIGEL_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
