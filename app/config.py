from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FastAPI Vertex AI Gemini Bridge"
    app_env: str = "development"
    port: int = 8000
    host: str = "0.0.0.0"

    # Google Cloud Vertex AI Configuration
    gcp_project_id: str = ""
    gcp_location: str = "us-central1"
    gemini_model: str = "gemini-1.5-flash"
    google_application_credentials: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()

